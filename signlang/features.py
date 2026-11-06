"""สกัดและแปลง landmark จาก MediaPipe Holistic ให้เป็นเวกเตอร์คุณลักษณะ

ข้อมูลแต่ละเฟรมเก็บเป็น
    points : (49, 2)  จุดร่างกาย 7 จุด + มือซ้าย 21 จุด + มือขวา 21 จุด
                      พิกัดถูกทำให้เป็นมาตรฐานเทียบกับไหล่ (จุดกึ่งกลางไหล่ = 0,
                      ความกว้างไหล่ = 1) จึงไม่ขึ้นกับตำแหน่ง/ระยะห่างจากกล้อง
    mask   : (2,)     1 = ตรวจพบมือซ้าย/ขวา, 0 = ไม่พบ
"""
import numpy as np

from . import config

# nose, ไหล่ซ้าย, ไหล่ขวา, ศอกซ้าย, ศอกขวา, ข้อมือซ้าย, ข้อมือขวา
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
N_POSE = len(POSE_IDX)
N_HAND = 21
N_POINTS = N_POSE + 2 * N_HAND
LH = slice(N_POSE, N_POSE + N_HAND)
RH = slice(N_POSE + N_HAND, N_POINTS)
POSE_WRISTS = (5, 6)          # ตำแหน่งข้อมือใน POSE_IDX
MIRROR_POSE_PAIRS = [(1, 2), (3, 4), (5, 6)]

FEAT_DIM = N_POINTS * 2 + 2 * N_HAND * 2 + 2


def _lm_array(landmarks, w, h, n):
    """แปลง landmark ของ MediaPipe เป็นพิกัดพิกเซล (n, 2)"""
    return np.array([[p.x * w, p.y * h] for p in landmarks.landmark[:n]], dtype=np.float32)


def frame_from_results(results, image_size):
    """แปลงผลลัพธ์ Holistic 1 เฟรม -> (points, mask, pose_vis) หรือ None ถ้าไม่พบร่างกาย"""
    if results.pose_landmarks is None:
        return None
    w, h = image_size
    pose_all = _lm_array(results.pose_landmarks, w, h, 33)
    vis = np.array([p.visibility for p in results.pose_landmarks.landmark], dtype=np.float32)

    l_sh, r_sh = pose_all[11], pose_all[12]
    center = (l_sh + r_sh) / 2.0
    scale = float(np.linalg.norm(l_sh - r_sh))
    if scale < 1e-3:
        return None

    points = np.zeros((N_POINTS, 2), dtype=np.float32)
    mask = np.zeros(2, dtype=np.float32)
    points[:N_POSE] = pose_all[POSE_IDX]
    if results.left_hand_landmarks is not None:
        points[LH] = _lm_array(results.left_hand_landmarks, w, h, N_HAND)
        mask[0] = 1
    if results.right_hand_landmarks is not None:
        points[RH] = _lm_array(results.right_hand_landmarks, w, h, N_HAND)
        mask[1] = 1

    points = (points - center) / scale
    if mask[0] == 0:
        points[LH] = 0
    if mask[1] == 0:
        points[RH] = 0
    return points, mask, vis[POSE_IDX]


def is_active(points, mask, pose_vis=None, thresh=config.ACTIVE_Y):
    """ผู้ใช้กำลังทำท่าหรือไม่: มีข้อมือ (จากมือหรือ pose) ยกสูงกว่าเกณฑ์"""
    ys = []
    if mask[0]:
        ys.append(points[LH][0, 1])
    if mask[1]:
        ys.append(points[RH][0, 1])
    for k in POSE_WRISTS:
        if pose_vis is None or pose_vis[k] > 0.5:
            ys.append(points[k, 1])
    return bool(ys) and min(ys) < thresh


def fill_missing_hands(points, mask):
    """เติมเฟรมที่มือหายไป (ตรวจจับพลาด) ด้วยการประมาณค่าเชิงเส้นจากเฟรมข้างเคียง"""
    points = points.copy()
    mask = mask.copy()
    n = len(points)
    t = np.arange(n)
    for hi, sl in enumerate((LH, RH)):
        present = mask[:, hi] > 0.5
        if present.all() or not present.any():
            continue
        idx = np.where(present)[0]
        # เติมเฉพาะช่วงระหว่างเฟรมที่พบมือ (ไม่ต่อขยายไปหัว/ท้าย)
        inner = (t > idx[0]) & (t < idx[-1]) & ~present
        if not inner.any():
            continue
        block = points[:, sl].reshape(n, -1)
        for d in range(block.shape[1]):
            block[inner, d] = np.interp(t[inner], idx, block[idx, d])
        points[:, sl] = block.reshape(n, N_HAND, 2)
        mask[inner, hi] = 1
    return points, mask


def resample(points, mask, length=config.SEQ_LEN, positions=None):
    """ปรับความยาวลำดับเป็น length เฟรม (ทำให้ไม่ขึ้นกับความเร็วในการทำท่า)"""
    n = len(points)
    if positions is None:
        positions = np.linspace(0, n - 1, length)
    i0 = np.floor(positions).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    a = (positions - i0)[:, None, None]
    p = points[i0] * (1 - a) + points[i1] * a
    m = mask[i0] * (1 - a[:, :, 0]) + mask[i1] * a[:, :, 0]
    # ถ้ามือหายในเฟรมใดเฟรมหนึ่ง ไม่ผสมกับ 0 ให้ใช้เฟรมที่มีมือแทน
    for hi, sl in enumerate((LH, RH)):
        only0 = (mask[i0, hi] > 0.5) & (mask[i1, hi] < 0.5)
        only1 = (mask[i0, hi] < 0.5) & (mask[i1, hi] > 0.5)
        p[only0, sl] = points[i0[only0], sl]
        p[only1, sl] = points[i1[only1], sl]
        m[only0 | only1, hi] = 1
    return p.astype(np.float32), m.astype(np.float32)


def mirror(points, mask):
    """สะท้อนซ้าย-ขวา (รองรับผู้ที่ถนัดมือซ้าย)"""
    p = points.copy()
    p[..., 0] *= -1
    for a, b in MIRROR_POSE_PAIRS:
        p[:, [a, b]] = p[:, [b, a]]
    lh = p[:, LH].copy()
    p[:, LH] = p[:, RH]
    p[:, RH] = lh
    return p, mask[:, ::-1].copy()


def to_features(points, mask):
    """(T, 49, 2) + (T, 2) -> (T, FEAT_DIM)

    ประกอบด้วย ตำแหน่งเทียบกับร่างกาย + รูปมือเทียบกับข้อมือ (บอกรูปร่างมือ) + การพบมือ
    """
    T = len(points)
    local = []
    for hi, sl in enumerate((LH, RH)):
        hand = points[:, sl]
        rel = hand - hand[:, :1]
        size = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=-1)[:, None, None]
        rel = np.where(size > 1e-4, rel / np.maximum(size, 1e-4), 0.0)
        rel *= mask[:, hi][:, None, None] > 0.5
        local.append(rel.reshape(T, -1))
    return np.concatenate([points.reshape(T, -1), *local, mask], axis=1).astype(np.float32)


def prepare_sequence(points, mask, length=config.SEQ_LEN):
    """ขั้นตอนเดียวกันทั้งตอนเทรนและตอนใช้งานจริง"""
    points, mask = fill_missing_hands(points, mask)
    points, mask = resample(points, mask, length)
    return to_features(points, mask)


def trim_active(active, pad=2, min_frames=6):
    """คืนค่า slice ของช่วงที่กำลังทำท่า (ใช้กับ GIF ให้เหมือนกับการจับท่าตอนเรียลไทม์)"""
    idx = np.where(active)[0]
    if len(idx) < min_frames:
        return slice(0, len(active))
    return slice(max(idx[0] - pad, 0), min(idx[-1] + pad + 1, len(active)))


def to_rate(points, mask, t, fps=config.BASE_FPS):
    """ปรับลำดับให้เป็นอัตราเฟรมคงที่ตามเวลาจริง (GIF 12.5 fps / กล้อง 15-30 fps -> 25 fps)"""
    t = np.asarray(t, dtype=np.float64)
    points, mask = fill_missing_hands(points, mask)
    n_new = max(int(round((t[-1] - t[0]) * fps)) + 1, 2)
    positions = np.interp(t[0] + np.arange(n_new) / fps, t, np.arange(len(t)))
    return resample(points, mask, n_new, positions=positions)
