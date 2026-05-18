"""
Connect-X Agent V9 - REBUILT FROM V5 (bàn 6x7, inarow=4)
---------------------------------------------------------
TƯ TƯỞNG:
  Quay lại V5 (bộ search đã chứng minh ổn định) và CHỈ bổ sung 3 cải tiến. Không thêm modes, không hybrid, không trick rủi ro.

Cải tiến so với V5:
  (1) BỎ MCTS HYBRID
      - Loại _playout / _mcts_verify. MCTS playout random với độ sâu lớn
        thường cho kết quả nhiễu, đôi khi override quyết định đúng của negamax.

  (2) FAST BITBOARD EVAL (kế thừa ý tưởng tốt từ V7)
      - V5 dùng _bb_to_grid() rồi _score_window(): mỗi leaf node phải dựng
        lại grid 42 phần tử và quét 69 windows trên list Python.
      - V9 dùng 69 winning-lines pre-computed dưới dạng bitmask, eval thuần
        bit-ops: nhanh hơn ~5x → search sâu hơn trong 2s.
      - Vẫn giữ Allis claim-even/odd-threat (đã hợp lý ở V5).

  (3) FIX BUG TT MATE-DISTANCE
      - V5/V7 lưu mate-score theo ply tuyệt đối từ root. Cùng một position
        được reach từ nhiều ply khác nhau → lookup trả score sai → đôi khi
        thấy thắng mà thực ra không phải, hoặc bỏ qua mate gần hơn.
      - V9 normalize: store `score ± ply`, load `score ∓ ply`.
        Đây là bug-fix có tính đúng đắn, không phải tuning.

  (4) MOVE ORDERING ANTI-TRAP TOÀN PHẦN
      - V5 chỉ check ô ngay trên cùng cột (above) — bỏ sót gift moves
        cho phép opp thắng ở cột khác.
      - V9 quét toàn bộ 7 cột: sau khi ta đi col c, opp có thắng được không?
        Nếu có → phạt -180k.
      - Thêm bonus +50k cho move tạo ≥ 2 winning replies (double-threat).

KHÔNG thay đổi:
  - Negamax + PVS, Symmetry-aware TT
  - Killer / history / counter
  - LMR + Aspiration window
  - Forced-move + Double-threat early detection
  - IID + Singular Extension (giữ nguyên ngưỡng V5)
  - Endgame exact solver
  - Opening: center cho 2 nước đầu

API: act(observation, configuration) -> int   (tương thích Kaggle).
"""

import time


# =====================================================================
#                         Bitboard (6x7, inarow=4)
# =====================================================================
HEIGHT = 6
WIDTH = 7
H1 = HEIGHT + 1
H2 = H1 + 1
SIZE = HEIGHT * WIDTH

FULL_MASK = 0
for _c in range(WIDTH):
    FULL_MASK |= ((1 << HEIGHT) - 1) << (_c * H1)
COLUMN_MASK = [((1 << HEIGHT) - 1) << (c * H1) for c in range(WIDTH)]
TOP_MASK = [1 << ((c * H1) + HEIGHT - 1) for c in range(WIDTH)]
BOT_MASK = [1 << (c * H1) for c in range(WIDTH)]


# =====================================================================
#               (2) PRE-COMPUTED 69 WINNING LINES (bitmask)
# =====================================================================
# 24 horizontal + 21 vertical + 12 diag \ + 12 diag /
# Mỗi line là (mask, lowest_rfb) — lowest_rfb dùng cho parity hint.

WINNING_LINES = []


def _init_winning_lines():
    global WINNING_LINES
    lines = []
    # Horizontal
    for rfb in range(HEIGHT):
        for start_col in range(WIDTH - 3):
            mask = 0
            for o in range(4):
                col = start_col + o
                mask |= 1 << (col * H1 + rfb)
            lines.append((mask, rfb))
    # Vertical
    for col in range(WIDTH):
        for start_rfb in range(HEIGHT - 3):
            mask = 0
            for o in range(4):
                mask |= 1 << (col * H1 + (start_rfb + o))
            lines.append((mask, start_rfb))
    # Diagonal \
    for start_rfb in range(HEIGHT - 1, 2, -1):
        for start_col in range(WIDTH - 3):
            mask = 0
            lowest = start_rfb
            for o in range(4):
                col = start_col + o
                rfb = start_rfb - o
                mask |= 1 << (col * H1 + rfb)
                if rfb < lowest:
                    lowest = rfb
            lines.append((mask, lowest))
    # Diagonal /
    for start_rfb in range(HEIGHT - 3):
        for start_col in range(WIDTH - 3):
            mask = 0
            for o in range(4):
                col = start_col + o
                rfb = start_rfb + o
                mask |= 1 << (col * H1 + rfb)
            lines.append((mask, start_rfb))
    WINNING_LINES = lines


_init_winning_lines()

CENTER_WEIGHTS = [max(0, 4 - abs(c - WIDTH // 2)) for c in range(WIDTH)]


# =====================================================================
#                     Basic Bitboard Operations
# =====================================================================

def _bb_alignment(pos):
    m = pos & (pos >> 1)
    if m & (m >> 2):
        return True
    m = pos & (pos >> H1)
    if m & (m >> (2 * H1)):
        return True
    m = pos & (pos >> HEIGHT)
    if m & (m >> (2 * HEIGHT)):
        return True
    m = pos & (pos >> H2)
    if m & (m >> (2 * H2)):
        return True
    return False


def _bb_play(position, mask, col):
    new_opp_position = position ^ mask
    new_mask = mask | (mask + BOT_MASK[col])
    return new_opp_position, new_mask


def _bb_move_bit(mask, col):
    return (mask + BOT_MASK[col]) & COLUMN_MASK[col]


def _bb_from_board(board, rows, columns, mark):
    position = 0
    mask = 0
    for col in range(columns):
        for row in range(rows):
            v = board[row * columns + col]
            if v == 0:
                continue
            bit = 1 << (col * H1 + (rows - 1 - row))
            mask |= bit
            if v == mark:
                position |= bit
    return position, mask


def _bb_winning_positions(pos):
    """Bitmask các ô mà nếu pos đặt 1 quân vào -> 4-in-a-row (chưa lọc empty)."""
    r = 0
    p = (pos << 1) & (pos << 2) & (pos << 3)
    r |= p
    p = (pos << H1) & (pos << (2 * H1))
    r |= p & (pos << (3 * H1))
    r |= p & (pos >> H1)
    p = (pos >> H1) & (pos >> (2 * H1))
    r |= p & (pos << H1)
    r |= p & (pos >> (3 * H1))
    s = HEIGHT
    p = (pos << s) & (pos << (2 * s))
    r |= p & (pos << (3 * s))
    r |= p & (pos >> s)
    p = (pos >> s) & (pos >> (2 * s))
    r |= p & (pos << s)
    r |= p & (pos >> (3 * s))
    s = H2
    p = (pos << s) & (pos << (2 * s))
    r |= p & (pos << (3 * s))
    r |= p & (pos >> s)
    p = (pos >> s) & (pos >> (2 * s))
    r |= p & (pos << s)
    r |= p & (pos >> (3 * s))
    return r & FULL_MASK


# =====================================================================
#                  Symmetry-aware TT key (giữ từ V5)
# =====================================================================

def _mirror_bb(bb):
    r = 0
    col_mask_full = (1 << H1) - 1
    for c in range(WIDTH):
        col_bits = (bb >> (c * H1)) & col_mask_full
        r |= col_bits << ((WIDTH - 1 - c) * H1)
    return r


def _canonical_key(position, mask):
    mpos = _mirror_bb(position)
    mmask = _mirror_bb(mask)
    if (mmask, mpos) < (mask, position):
        return (mpos, mmask), True
    return (position, mask), False


# =====================================================================
#       (2) FAST BITBOARD EVAL — 69 winning lines + Allis parity
# =====================================================================

def _get_playable_mask(mask):
    playable = 0
    for col in range(WIDTH):
        if (mask & TOP_MASK[col]) == 0:
            playable |= _bb_move_bit(mask, col)
    return playable


def _count_playable_wins(pos, mask, limit=7):
    win_bits = _bb_winning_positions(pos) & (FULL_MASK ^ mask)
    if not win_bits:
        return 0
    cnt = 0
    for col in range(WIDTH):
        if (mask & TOP_MASK[col]) != 0:
            continue
        if _bb_move_bit(mask, col) & win_bits:
            cnt += 1
            if cnt >= limit:
                return cnt
    return cnt


def _move_allows_opp_win(position, mask, col):
    new_opp_pos, new_mask = _bb_play(position, mask, col)
    return _count_playable_wins(new_opp_pos, new_mask, 1) > 0


def _evaluate(my_pos, opp_pos, mask, mover_is_first):
    """Eval từ góc nhìn side-to-move. Dùng bit ops trên 69 winning lines
    + Allis claim-even / odd-threat parity. Không convert sang grid."""
    empty_mask = FULL_MASK ^ mask
    playable = _get_playable_mask(mask)

    score = 0

    # Center control (giữ nhẹ, không đậm như V7)
    for col in range(WIDTH):
        col_mask = COLUMN_MASK[col]
        my_col = (my_pos & col_mask).bit_count()
        opp_col = (opp_pos & col_mask).bit_count()
        score += (my_col - opp_col) * CENTER_WEIGHTS[col]

    # Parity tốt cho side-to-move (theo Allis)
    my_good_parity = 0 if mover_is_first else 1

    # Quét 69 winning lines bằng bit ops
    for line_mask, lowest_rfb in WINNING_LINES:
        my_count = (my_pos & line_mask).bit_count()
        opp_count = (opp_pos & line_mask).bit_count()

        # Dead line - có cả 2 bên -> bỏ qua
        if my_count > 0 and opp_count > 0:
            continue

        empty_in_line = 4 - my_count - opp_count
        playable_empty = (line_mask & empty_mask & playable).bit_count()

        # MY threats
        if my_count == 4:
            score += 100000
        elif my_count == 3 and empty_in_line == 1:
            if playable_empty >= 1:
                score += 230
                if lowest_rfb % 2 == my_good_parity:
                    score += 30
            else:
                score += 55
        elif my_count == 2 and empty_in_line == 2:
            if playable_empty >= 1:
                score += 22
            else:
                score += 6
        elif my_count == 1 and empty_in_line == 3:
            score += 2

        # OPP threats
        if opp_count == 4:
            score -= 100000
        elif opp_count == 3 and empty_in_line == 1:
            if playable_empty >= 1:
                score -= 280
                if lowest_rfb % 2 != my_good_parity:
                    score -= 40
            else:
                score -= 70
        elif opp_count == 2 and empty_in_line == 2:
            if playable_empty >= 1:
                score -= 20
            else:
                score -= 5
        elif opp_count == 1 and empty_in_line == 3:
            score -= 2

    # Allis zugzwang per-column (giữ tinh thần V5)
    my_threats = _bb_winning_positions(my_pos) & empty_mask
    opp_threats = _bb_winning_positions(opp_pos) & empty_mask

    for col in range(WIDTH):
        base = col * H1
        my_lowest = -1
        opp_lowest = -1
        for rfb in range(HEIGHT):
            bit = 1 << (base + rfb)
            if my_lowest < 0 and (my_threats & bit):
                my_lowest = rfb
            if opp_lowest < 0 and (opp_threats & bit):
                opp_lowest = rfb

        if my_lowest >= 0:
            good = (my_lowest % 2) == my_good_parity
            blocked = opp_lowest >= 0 and opp_lowest < my_lowest
            if not blocked and good:
                score += 45
            elif good:
                score += 15
            elif not blocked:
                score += 12
            else:
                score += 4

        if opp_lowest >= 0:
            good = (opp_lowest % 2) != my_good_parity
            blocked = my_lowest >= 0 and my_lowest < opp_lowest
            if not blocked and good:
                score -= 45
            elif good:
                score -= 15
            elif not blocked:
                score -= 12
            else:
                score -= 4

    return score


# =====================================================================
#                  Search: Negamax + PVS + Symmetry TT
# =====================================================================

_TT = {}
_TT_MAX = 1_500_000
FLAG_EXACT = 0
FLAG_LOWER = 1
FLAG_UPPER = -1

_killers = {}
_history = {}
_counter = {}

_nodes = 0
INF = 10_000_000
WIN_BASE = 1_000_000
MATE_THRESHOLD = WIN_BASE - 1000


class _Timeout(Exception):
    pass


# ---------- (3) TT mate-distance normalization ----------

def _tt_store_score(score, ply):
    """score là RELATIVE-TO-ROOT. Lưu RELATIVE-TO-NODE để reuse được."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _tt_load_score(stored, ply):
    """Đảo ngược: từ RELATIVE-TO-NODE về RELATIVE-TO-ROOT."""
    if stored >= MATE_THRESHOLD:
        return stored - ply
    if stored <= -MATE_THRESHOLD:
        return stored + ply
    return stored


# ---------- (4) Move ordering with FULL anti-trap ----------

def _ordered_moves(position, mask, depth, last_col=-1):
    """Win > block > double-threat-bonus > killer/counter > history > center.
    Phạt mạnh "gift move": nước cho phép opp thắng ngay ở bất kỳ cột nào."""
    killers = _killers.get(depth)
    counter_reply = _counter.get(last_col) if last_col >= 0 else None
    center = WIDTH // 2
    opp = position ^ mask
    opp_win_bits = _bb_winning_positions(opp) & (FULL_MASK ^ mask)
    my_win_bits = _bb_winning_positions(position) & (FULL_MASK ^ mask)

    out = []
    for col in range(WIDTH):
        if (mask & TOP_MASK[col]) != 0:
            continue

        s = -abs(col - center) * 3
        mv = _bb_move_bit(mask, col)

        # Win move
        if mv & my_win_bits:
            s += 1_000_000

        # Block opp's immediate win
        if mv & opp_win_bits:
            s += 500_000

        # FULL anti-trap: sau khi mình đi, opp có thắng ngay ở bất kỳ cột nào không?
        gift = _move_allows_opp_win(position, mask, col)

        if gift:
            s -= 180_000
        else:
            # Bonus tạo double-threat (chỉ tính khi không gift)
            new_msk = mask | mv
            new_my = position | mv
            if _count_playable_wins(new_my, new_msk, 2) >= 2:
                s += 50_000

        if killers and col in killers:
            s += 10_000
        if col == counter_reply:
            s += 8_000
        s += _history.get(col, 0)

        out.append((s, col))

    out.sort(reverse=True)
    return [c for _, c in out]


# Search constants (giữ giá trị V5)
LMR_MIN_DEPTH = 3
LMR_START_INDEX = 3
ASPIRATION_DELTA = 40
ASPIRATION_MIN_DEPTH = 4
ENDGAME_THRESHOLD = 16

SINGULAR_MIN_DEPTH = 8
SINGULAR_MARGIN = 80


def _double_threat_move(position, mask):
    """Trả về cột tạo ≥2 immediate threats mà opp không thắng phản đòn."""
    for col in range(WIDTH):
        if (mask & TOP_MASK[col]) != 0:
            continue
        mv = _bb_move_bit(mask, col)
        new_mask = mask | mv
        new_my = position | mv

        if _move_allows_opp_win(position, mask, col):
            continue

        if _count_playable_wins(new_my, new_mask, 2) >= 2:
            return col
    return -1


def _negamax(position, mask, depth, alpha, beta, ply, mover_is_first, deadline,
             endgame_mode=False, last_col=-1):
    global _nodes
    _nodes += 1
    if (_nodes & 1023) == 0 and time.time() > deadline:
        raise _Timeout

    opp = position ^ mask
    if _bb_alignment(opp):
        return -WIN_BASE + ply
    if mask.bit_count() == SIZE:
        return 0

    # Immediate-win scan
    for col in range(WIDTH):
        if (mask & TOP_MASK[col]) != 0:
            continue
        mv = _bb_move_bit(mask, col)
        if _bb_alignment(position | mv):
            return WIN_BASE - ply

    # Forced-move detection
    opp_threats = []
    for col in range(WIDTH):
        if (mask & TOP_MASK[col]) != 0:
            continue
        mv = _bb_move_bit(mask, col)
        if _bb_alignment(opp | mv):
            opp_threats.append(col)
            if len(opp_threats) > 1:
                break

    if len(opp_threats) >= 2:
        return -WIN_BASE + ply + 2
    if len(opp_threats) == 1:
        forced_col = opp_threats[0]
        new_opp_pos, new_mask = _bb_play(position, mask, forced_col)
        next_depth = depth - 1 if depth >= 1 else 0
        return -_negamax(new_opp_pos, new_mask, next_depth, -beta, -alpha,
                         ply + 1, not mover_is_first, deadline,
                         endgame_mode, forced_col)

    # Double-threat early detection (PV-node, depth >= 2)
    if (beta - alpha) > 1 and depth >= 2:
        dt_col = _double_threat_move(position, mask)
        if dt_col >= 0:
            score = WIN_BASE - ply - 2
            key, mirrored = _canonical_key(position, mask)
            store_col = (WIDTH - 1 - dt_col) if mirrored else dt_col
            if len(_TT) < _TT_MAX:
                existing = _TT.get(key)
                if existing is None or existing[0] <= depth:
                    _TT[key] = (depth, FLAG_LOWER,
                                _tt_store_score(score, ply), store_col)
            return score

    if depth == 0:
        return 0 if endgame_mode else _evaluate(position, opp, mask, mover_is_first)

    # TT lookup with mate-distance normalization
    key, mirrored = _canonical_key(position, mask)
    alpha_orig = alpha
    tt = _TT.get(key)
    tt_best = None
    tt_real_score = None
    if tt is not None:
        tt_d, tt_flag, tt_stored, tt_stored_best = tt
        tt_best = (WIDTH - 1 - tt_stored_best) if (mirrored and tt_stored_best is not None) else tt_stored_best
        tt_real_score = _tt_load_score(tt_stored, ply)
        if tt_d >= depth:
            if tt_flag == FLAG_EXACT:
                return tt_real_score
            if tt_flag == FLAG_LOWER and tt_real_score > alpha:
                alpha = tt_real_score
            elif tt_flag == FLAG_UPPER and tt_real_score < beta:
                beta = tt_real_score
            if alpha >= beta:
                return tt_real_score

    # IID
    if tt_best is None and depth >= 4 and (beta - alpha) > 1 and not endgame_mode:
        _negamax(position, mask, depth - 2, alpha, beta, ply,
                 mover_is_first, deadline, endgame_mode, last_col)
        iid_tt = _TT.get(key)
        if iid_tt is not None:
            iid_best = iid_tt[3]
            tt_best = (WIDTH - 1 - iid_best) if (mirrored and iid_best is not None) else iid_best

    moves = _ordered_moves(position, mask, depth, last_col)
    if tt_best is not None and tt_best in moves:
        moves.remove(tt_best)
        moves.insert(0, tt_best)

    # Singular Extension
    singular_extension = 0
    if (tt is not None
            and tt_best is not None
            and tt[0] >= depth - 2
            and (tt[1] == FLAG_EXACT or tt[1] == FLAG_LOWER)
            and depth >= SINGULAR_MIN_DEPTH
            and (beta - alpha) > 1
            and not endgame_mode
            and tt_real_score is not None
            and abs(tt_real_score) < MATE_THRESHOLD):
        verify_bound = tt_real_score - SINGULAR_MARGIN
        reduced = max(1, depth // 2)
        singular = True
        for col in moves:
            if col == tt_best:
                continue
            new_opp_pos, new_mask = _bb_play(position, mask, col)
            if _count_playable_wins(new_opp_pos, new_mask, 1) > 0:
                score = -WIN_BASE + ply + 1
            else:
                score = -_negamax(new_opp_pos, new_mask, reduced,
                                  -verify_bound - 1, -verify_bound,
                                  ply + 1, not mover_is_first, deadline,
                                  endgame_mode, col)
            if score >= verify_bound:
                singular = False
                break
        if singular:
            singular_extension = 1

    best_score = -INF
    best_col = moves[0] if moves else 0
    killers_here = _killers.get(depth) or []
    opp_win_mask_lmr = _bb_winning_positions(opp) & (FULL_MASK ^ mask)

    for i, col in enumerate(moves):
        new_opp_pos, new_mask = _bb_play(position, mask, col)
        if _count_playable_wins(new_opp_pos, new_mask, 1) > 0:
            score = -WIN_BASE + ply + 1
        else:
            # LMR
            reduction = 0
            mv_bit_lmr = _bb_move_bit(mask, col)
            blocks_opp_threat = bool(mv_bit_lmr & opp_win_mask_lmr)
            if (not endgame_mode
                    and i >= LMR_START_INDEX
                    and depth >= LMR_MIN_DEPTH
                    and col not in killers_here
                    and not blocks_opp_threat
                    and abs(best_score) < MATE_THRESHOLD):
                reduction = 1 if i < 5 else 2
                if reduction >= depth:
                    reduction = depth - 1

            ext_here = singular_extension if (i == 0 and col == tt_best) else 0

            if i == 0:
                score = -_negamax(new_opp_pos, new_mask, depth - 1 + ext_here,
                                  -beta, -alpha, ply + 1, not mover_is_first,
                                  deadline, endgame_mode, col)
            else:
                score = -_negamax(new_opp_pos, new_mask, depth - 1 - reduction,
                                  -alpha - 1, -alpha, ply + 1, not mover_is_first,
                                  deadline, endgame_mode, col)
                if reduction > 0 and score > alpha:
                    score = -_negamax(new_opp_pos, new_mask, depth - 1,
                                      -alpha - 1, -alpha, ply + 1, not mover_is_first,
                                      deadline, endgame_mode, col)
                if alpha < score < beta:
                    score = -_negamax(new_opp_pos, new_mask, depth - 1,
                                      -beta, -score, ply + 1, not mover_is_first,
                                      deadline, endgame_mode, col)

        if score > best_score:
            best_score = score
            best_col = col
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            ks = _killers.setdefault(depth, [])
            if col not in ks:
                ks.insert(0, col)
                if len(ks) > 2:
                    ks.pop()
            _history[col] = _history.get(col, 0) + depth * depth
            if last_col >= 0:
                _counter[last_col] = col
            break

    if best_score <= alpha_orig:
        flag = FLAG_UPPER
    elif best_score >= beta:
        flag = FLAG_LOWER
    else:
        flag = FLAG_EXACT

    if len(_TT) >= _TT_MAX:
        _TT.clear()

    store_best = (WIDTH - 1 - best_col) if mirrored else best_col
    stored_score = _tt_store_score(best_score, ply)
    existing = _TT.get(key)
    if existing is None or existing[0] <= depth:
        _TT[key] = (depth, flag, stored_score, store_best)
    return best_score


def _root_iter(position, mask, depth, alpha, beta, mover_is_first, deadline,
               hint_col, endgame_mode, last_opp_col=-1):
    moves = _ordered_moves(position, mask, depth, last_opp_col)
    if hint_col in moves:
        moves.remove(hint_col)
        moves.insert(0, hint_col)
    local_best_col = hint_col if moves and hint_col in moves else moves[0]
    local_best_score = -INF
    a = alpha
    for i, col in enumerate(moves):
        new_opp_pos, new_mask = _bb_play(position, mask, col)
        if _count_playable_wins(new_opp_pos, new_mask, 1) > 0:
            score = -WIN_BASE + 1
        else:
            if i == 0:
                score = -_negamax(new_opp_pos, new_mask, depth - 1, -beta, -a,
                                  1, not mover_is_first, deadline, endgame_mode, col)
            else:
                score = -_negamax(new_opp_pos, new_mask, depth - 1, -a - 1, -a,
                                  1, not mover_is_first, deadline, endgame_mode, col)
                if a < score < beta:
                    score = -_negamax(new_opp_pos, new_mask, depth - 1, -beta, -score,
                                      1, not mover_is_first, deadline, endgame_mode, col)
        if score > local_best_score:
            local_best_score = score
            local_best_col = col
        if score > a:
            a = score
        if a >= beta:
            break
    return local_best_score, local_best_col


def _root_search(position, mask, mover_is_first, deadline, endgame_mode=False,
                 last_opp_col=-1):
    global _nodes
    center = WIDTH // 2
    valid = [c for c in range(WIDTH) if (mask & TOP_MASK[c]) == 0]
    if not valid:
        return 0, 0, 0
    best_col = sorted(valid, key=lambda c: abs(c - center))[0]
    best_score = 0
    depth_reached = 0
    prev_score = 0

    for depth in range(1, SIZE + 1):
        if time.time() > deadline:
            break
        use_aspiration = depth >= ASPIRATION_MIN_DEPTH and not endgame_mode
        if use_aspiration:
            delta = ASPIRATION_DELTA
            alpha = prev_score - delta
            beta = prev_score + delta
        else:
            alpha, beta = -INF, INF
            delta = INF

        try:
            while True:
                score, col = _root_iter(position, mask, depth, alpha, beta,
                                        mover_is_first, deadline, best_col,
                                        endgame_mode, last_opp_col)
                if use_aspiration and score <= alpha and alpha > -INF + 10:
                    delta *= 2
                    alpha = prev_score - delta
                    if alpha < -INF + 10:
                        alpha = -INF
                    continue
                if use_aspiration and score >= beta and beta < INF - 10:
                    delta *= 2
                    beta = prev_score + delta
                    if beta > INF - 10:
                        beta = INF
                    continue
                break
        except _Timeout:
            break

        best_col = col
        best_score = score
        depth_reached = depth
        prev_score = score

        if best_score >= MATE_THRESHOLD or best_score <= -MATE_THRESHOLD:
            break

    return best_col, best_score, depth_reached


# =====================================================================
#                       Opening book (giữ tối thiểu)
# =====================================================================

def _opening_move(board, mark, moves_played):
    center = WIDTH // 2
    if moves_played == 0:
        return center
    if moves_played == 1:
        bottom_center = (HEIGHT - 1) * WIDTH + center
        if board[bottom_center] == 0:
            return center
        return center + 1
    return None


# =====================================================================
#                  Fallback an toàn cho config khác
# =====================================================================

def _drop_row_list(board, col, rows, columns):
    for r in range(rows - 1, -1, -1):
        if board[r * columns + col] == 0:
            return r
    return None


def _win_at_list(board, mark, r, c, rows, columns, inarow):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while 0 <= rr < rows and 0 <= cc < columns and board[rr * columns + cc] == mark:
                cnt += 1
                if cnt >= inarow:
                    return True
                rr += s * dr
                cc += s * dc
    return False


def _fallback_act(board, rows, columns, inarow, mark):
    opp = 2 if mark == 1 else 1
    valid = [c for c in range(columns) if board[c] == 0]
    if not valid:
        return 0
    for c in valid:
        r = _drop_row_list(board, c, rows, columns)
        if r is None:
            continue
        i = r * columns + c
        board[i] = mark
        if _win_at_list(board, mark, r, c, rows, columns, inarow):
            board[i] = 0
            return c
        board[i] = 0
    for c in valid:
        r = _drop_row_list(board, c, rows, columns)
        if r is None:
            continue
        i = r * columns + c
        board[i] = opp
        if _win_at_list(board, opp, r, c, rows, columns, inarow):
            board[i] = 0
            return c
        board[i] = 0
    return sorted(valid, key=lambda c: abs(c - columns // 2))[0]


# =====================================================================
#                              act()
# =====================================================================

def _reset_per_move():
    _killers.clear()
    _history.clear()
    _counter.clear()


def _cell_playable(board, col):
    for r in range(HEIGHT - 1, -1, -1):
        if board[r * WIDTH + col] == 0:
            return True
    return False


def act(observation, configuration):
    board = list(observation.board)
    columns = configuration.columns
    rows = configuration.rows
    inarow = configuration.inarow
    mark = observation.mark

    if rows != HEIGHT or columns != WIDTH or inarow != 4:
        return _fallback_act(board, rows, columns, inarow, mark)

    moves_played = sum(1 for v in board if v != 0)

    op = _opening_move(board, mark, moves_played)
    if op is not None and _cell_playable(board, op):
        return op

    position, mask = _bb_from_board(board, rows, columns, mark)

    # Priority 1: thắng ngay
    for col in range(WIDTH):
        if (mask & TOP_MASK[col]) != 0:
            continue
        mv = _bb_move_bit(mask, col)
        if _bb_alignment(position | mv):
            return col

    # Priority 2: chặn opp thắng ngay
    opp_pos = position ^ mask
    for col in range(WIDTH):
        if (mask & TOP_MASK[col]) != 0:
            continue
        mv = _bb_move_bit(mask, col)
        if _bb_alignment(opp_pos | mv):
            return col

    # Priority 3: táº¡o double-threat an toÃ n
    dt_col = _double_threat_move(position, mask)
    if dt_col >= 0:
        return int(dt_col)

    _reset_per_move()

    # Time budget: 2.0 - 0.35 = 1.65s (giống V5, ổn định)
    timeout = getattr(configuration, "timeout", 2.0)
    time_budget = timeout - 0.35
    if time_budget < 0.15:
        time_budget = max(0.10, timeout * 0.6)
    start = time.time()
    deadline = start + time_budget

    mover_is_first = (moves_played % 2 == 0)
    empty_count = SIZE - mask.bit_count()

    global _nodes
    _nodes = 0

    # Endgame exact solver
    if empty_count <= ENDGAME_THRESHOLD:
        _TT.clear()
        best_col, _, _ = _root_search(
            position, mask, mover_is_first, deadline, endgame_mode=True,
        )
        valid = [c for c in range(WIDTH) if (mask & TOP_MASK[c]) == 0]
        if best_col not in valid:
            best_col = sorted(valid, key=lambda c: abs(c - WIDTH // 2))[0]
        return int(best_col)

    if len(_TT) > _TT_MAX:
        _TT.clear()

    best_col, _, _ = _root_search(
        position, mask, mover_is_first, deadline, endgame_mode=False,
    )

    valid = [c for c in range(WIDTH) if (mask & TOP_MASK[c]) == 0]
    if best_col not in valid:
        best_col = sorted(valid, key=lambda c: abs(c - WIDTH // 2))[0]
    return int(best_col)
