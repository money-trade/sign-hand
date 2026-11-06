"""Data augmentation: สร้างตัวอย่างฝึกจำนวนมากจากวิดีโอต้นแบบไม่กี่ไฟล์

จำลองความแตกต่างระหว่างผู้ใช้จริง: ความเร็ว จังหวะ ขนาดตัว มุมกล้อง ตำแหน่ง
ความสั่นของมือ การตรวจจับมือพลาดบางเฟรม และผู้ถนัดมือซ้าย

สำหรับการจับท่าแบบต่อเนื่อง (sliding window) ยังสร้าง
  - ตัวอย่างบวกที่มีส่วนท้าย/ต้นของท่าข้างเคียงติดมาเล็กน้อย (หน้าต่างไม่ตรงพอดี)
  - ตัวอย่างลบ (_none) : ท่าไม่ครบ, หน้าต่างคร่อมสองท่า, มือขยับมั่ว ๆ
"""
import numpy as np

from . import config
from .features import (LH, N_HAND, N_POINTS, N_POSE, RH, fill_missing_hands, mirror,
                       resample, to_features)


def _time_warp_positions(n, length, rng, strength=0.35):
    """ตำแหน่งสุ่มเฟรมแบบไม่เชิงเส้น (บางช่วงทำเร็ว บางช่วงทำช้า)"""
    knots = rng.uniform(1 - strength, 1 + strength, size=5)
    speed = np.interp(np.linspace(0, 4, length), np.arange(5), knots)
    pos = np.concatenate([[0], np.cumsum(speed)[:-1]])
    return pos / pos[-1] * (n - 1)


def _rotate_hand(points, sl, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    wrist = points[:, sl][:, :1]
    points[:, sl] = (points[:, sl] - wrist) @ R.T + wrist


def join(parts, rng):
    """ต่อหลายช่วงท่าเข้าด้วยกัน โดยเติมเฟรมเปลี่ยนท่าแบบค่อย ๆ เลื่อน (เหมือนทำท่าต่อเนื่อง)"""
    out_p, out_m = [parts[0][0]], [parts[0][1]]
    for p, m in parts[1:]:
        a_p, a_m = out_p[-1][-1], out_m[-1][-1]
        b_p, b_m = p[0], m[0]
        k = int(rng.integers(2, 6))
        for i in range(1, k + 1):
            w = i / (k + 1)
            fp = a_p * (1 - w) + b_p * w
            fm = np.where((a_m > 0.5) & (b_m > 0.5), 1.0, 0.0).astype(np.float32)
            fp[LH] *= fm[0]
            fp[RH] *= fm[1]
            out_p.append(fp[None])
            out_m.append(fm[None])
        out_p.append(p)
        out_m.append(m)
    return np.concatenate(out_p), np.concatenate(out_m)


def with_context(p, m, others, rng, max_frac=0.2):
    """ติดส่วนท้ายของท่าก่อนหน้า/ส่วนต้นของท่าถัดไปมาเล็กน้อย (<= 20% ของความยาวท่า)"""
    n = len(p)
    parts = []
    if others and rng.random() < 0.5:
        q, qm = others[rng.integers(len(others))]
        k = int(n * rng.uniform(0.05, max_frac))
        if k >= 1:
            parts.append((q[-k:], qm[-k:]))
    parts.append((p, m))
    if others and rng.random() < 0.5:
        q, qm = others[rng.integers(len(others))]
        k = int(n * rng.uniform(0.05, max_frac))
        if k >= 1:
            parts.append((q[:k], qm[:k]))
    return join(parts, rng) if len(parts) > 1 else (p, m)


def partial_sign(p, m, rng):
    """ท่าไม่ครบ: ช่วงต่อเนื่อง 20-55% ของท่า"""
    n = len(p)
    k = max(4, int(n * rng.uniform(0.2, 0.55)))
    s = int(rng.integers(0, max(1, n - k + 1)))
    return p[s:s + k], m[s:s + k]


def straddle(a, b, rng):
    """หน้าต่างคร่อมสองท่า: ท้ายท่า A 25-60% + ต้นท่า B 25-60%"""
    (pa, ma), (pb, mb) = a, b
    ka = max(2, int(len(pa) * rng.uniform(0.25, 0.6)))
    kb = max(2, int(len(pb) * rng.uniform(0.25, 0.6)))
    return join([(pa[-ka:], ma[-ka:]), (pb[:kb], mb[:kb])], rng)


def augment(points, mask, rng, length=config.SEQ_LEN, allow_mirror=True, crop=0.12):
    """คืนค่า features (length, FEAT_DIM) ของตัวอย่างที่ถูกสุ่มดัดแปลง 1 ตัวอย่าง"""
    points = points.copy()
    mask = mask.copy()
    n = len(points)

    # 1) ตัดหัว/ท้ายแบบสุ่ม (ผู้ใช้อาจเริ่ม/จบท่าเร็วหรือช้ากว่า)
    if n > 12 and crop > 0:
        s = rng.integers(0, max(1, int(n * crop)))
        e = n - rng.integers(0, max(1, int(n * crop)))
        points, mask = points[s:e], mask[s:e]
        n = len(points)

    # 2) มือหายบางเฟรม (ตรวจจับพลาด) -> แล้วเติมกลับแบบเดียวกับตอนใช้งานจริง
    if rng.random() < 0.6:
        drop = rng.random((n, 2)) < rng.uniform(0.05, 0.2)
        mask = mask * ~drop
        points[:, LH] *= mask[:, 0, None, None]
        points[:, RH] *= mask[:, 1, None, None]
    points, mask = fill_missing_hands(points, mask)

    # 3) ความเร็ว/จังหวะ
    pos = _time_warp_positions(n, length, rng)
    points, mask = resample(points, mask, length, positions=pos)

    # 4) สะท้อนซ้าย-ขวา
    if allow_mirror and rng.random() < 0.3:
        points, mask = mirror(points, mask)

    # 5) หมุนรูปมือรอบข้อมือเล็กน้อย
    for sl in (LH, RH):
        _rotate_hand(points, sl, rng.normal(0, 0.12))

    # 6) การแปลงเชิงพื้นที่ทั้งตัว: ขนาด หมุน เฉือน เลื่อน
    ang = rng.normal(0, 0.12)
    c, s = np.cos(ang), np.sin(ang)
    sx, sy = rng.uniform(0.85, 1.15, size=2)
    shear = rng.normal(0, 0.05)
    A = np.array([[c, -s], [s, c]]) @ np.array([[sx, shear], [0, sy]])
    points = points @ A.T.astype(np.float32)
    points += rng.normal(0, 0.08, size=2).astype(np.float32)

    # 7) สัญญาณรบกวนรายจุด + การสั่นของมือทั้งมือ
    points += rng.normal(0, 0.015, size=points.shape).astype(np.float32)
    for sl in (LH, RH):
        points[:, sl] += rng.normal(0, 0.03, size=(length, 1, 2)).astype(np.float32)

    points[:, LH] *= mask[:, 0, None, None] > 0.5
    points[:, RH] *= mask[:, 1, None, None] > 0.5
    return to_features(points, mask)


def synth_random_motion(sources, rng, length=config.SEQ_LEN, refs=None, min_dist=0.45):
    """มือขยับไปมาแบบสุ่มที่ไม่ตรงกับท่าใดเลย

    sources: list ของ (points, mask) ใช้ยืมรูปร่างลำตัวและรูปมือจริง
    refs   : trajectory ข้อมือของแต่ละคำ (length, 2) เพื่อกันไม่ให้สุ่มไปซ้ำกับท่าจริง
    """
    for _ in range(50):
        p_src, m_src = sources[rng.integers(len(sources))]
        body = p_src[rng.integers(len(p_src)), :N_POSE]
        points = np.zeros((length, N_POINTS, 2), dtype=np.float32)
        points[:, :N_POSE] = body
        mask = np.zeros((length, 2), dtype=np.float32)

        hands_present = [i for i in range(2) if m_src[:, i].any()]
        if not hands_present:
            continue
        use = [h for h in hands_present if rng.random() < 0.7] or [hands_present[0]]
        t = np.linspace(0, 1, length)[:, None]
        wrist_traj = None
        for hi in use:
            sl = LH if hi == 0 else RH
            frames = np.where(m_src[:, hi] > 0.5)[0]
            shape_a = p_src[rng.choice(frames), sl]
            shape_b = p_src[rng.choice(frames), sl]
            shape_a = shape_a - shape_a[:1]
            shape_b = shape_b - shape_b[:1]
            # trajectory แบบ Bezier สุ่ม (ทั้งแบบนิ่ง ๆ และแบบเคลื่อนที่)
            ctrl = rng.uniform([-1.6, -1.2], [1.6, 1.7], size=(4, 2))
            if rng.random() < 0.3:
                ctrl = ctrl[:1] + rng.normal(0, 0.08, size=(4, 2))
            b = ((1 - t) ** 3 * ctrl[0] + 3 * (1 - t) ** 2 * t * ctrl[1]
                 + 3 * (1 - t) * t ** 2 * ctrl[2] + t ** 3 * ctrl[3])
            mix = t[:, :, None]
            shape = shape_a[None] * (1 - mix) + shape_b[None] * mix
            points[:, sl] = shape + b[:, None, :]
            points[:, N_POSE - 2 + hi] = b  # ข้อมือของ pose ตามมือ
            mask[:, hi] = 1
            if wrist_traj is None or hi == 1:
                wrist_traj = b

        if refs is not None and wrist_traj is not None:
            d = min(np.linalg.norm(wrist_traj - r, axis=-1).mean() for r in refs)
            if d < min_dist:
                continue
        points += rng.normal(0, 0.015, size=points.shape).astype(np.float32)
        points[:, LH] *= mask[:, 0, None, None]
        points[:, RH] *= mask[:, 1, None, None]
        return to_features(points, mask)
    return None


def wrist_reference(points, mask, length=config.SEQ_LEN):
    """trajectory ข้อมือมือหลักของท่าต้นแบบ (ใช้กันการสุ่มคลาส none ไปทับท่าจริง)"""
    p, m = fill_missing_hands(points, mask)
    p, m = resample(p, m, length)
    hi = int(np.argmax(m.sum(0)))
    return p[:, N_POSE + hi * N_HAND]
