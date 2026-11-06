"""ขั้นที่ 1: สกัด landmark จากชุดข้อมูลวิดีโอ/GIF ภาษามือ

    python extract_landmarks.py

อ่านทุกไฟล์ .gif .mp4 .avi .mov .mkv .webm ในโฟลเดอร์ชุดข้อมูล
ชื่อไฟล์ = คำศัพท์ (1 ไฟล์ = 1 คำ) เช่น กิน.gif, หมอ.mp4, หมอ_2.mp4 (หลายตัวอย่างของคำเดียวกัน)
บันทึกเป็น data/landmarks/<ชื่อไฟล์>.npz
"""
import cv2
import numpy as np
from PIL import Image, ImageSequence

from signlang import config
from signlang.features import is_active
from signlang.tracker import HolisticTracker


def label_of(stem):
    """'หมอ_2' -> 'หมอ'"""
    head, _, tail = stem.rpartition("_")
    return head if head and tail.isdigit() else stem


MAX_CLIP_SEC = 8.0   # 1 ไฟล์ = 1 คำ คลิปที่ยาวกว่านี้ไม่ใช่คลิปคำศัพท์


def is_word_name(stem):
    """ชื่อไฟล์ต้องเป็นคำศัพท์ (ข้ามไฟล์อย่าง '2026-10-04 20-51-41.mp4' ที่เป็นวิดีโอบันทึกหน้าจอ)"""
    label = label_of(stem)
    return bool(label) and not any(ch.isdigit() or ch.isspace() for ch in label)


def iter_video(path):
    """อ่านเฟรมทีละเฟรม -> (RGB, เวลาเป็นวินาที) ไม่โหลดทั้งไฟล์เข้าหน่วยความจำ"""
    if path.suffix.lower() == ".gif":
        t = 0.0
        with Image.open(path) as im:
            for f in ImageSequence.Iterator(im):
                yield np.array(f.convert("RGB")), t
                t += (f.info.get("duration") or 40) / 1000.0
        return
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), i / fps
        i += 1
    cap.release()


def video_duration(path):
    if path.suffix.lower() == ".gif":
        with Image.open(path) as im:
            return sum((f.info.get("duration") or 40) for f in ImageSequence.Iterator(im)) / 1000.0
    cap = cv2.VideoCapture(str(path))
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25)
    cap.release()
    return d


def read_gif(path):
    return [f for f, _ in iter_video(path)]


def track_frames(path, static):
    points, masks, active, ts = [], [], [], []
    with HolisticTracker(static_image_mode=static, model_complexity=2) as tracker:
        for rgb, t in iter_video(path):
            _, fr = tracker.process(rgb.copy())
            if fr is None:
                continue
            p, m, vis = fr
            points.append(p)
            masks.append(m)
            active.append(is_active(p, m, vis))
            ts.append(t)
    return np.array(points), np.array(masks), np.array(active), np.array(ts)


def main():
    config.LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in config.GIF_DIR.iterdir() if p.suffix.lower() in config.VIDEO_EXTS)
    if not files:
        raise SystemExit(f"ไม่พบไฟล์วิดีโอ/GIF ใน {config.GIF_DIR}")

    for path in files:
        if not is_word_name(path.stem):
            print(f"{path.name}: ข้าม — ชื่อไฟล์ไม่ใช่คำศัพท์ (ตั้งชื่อเป็นคำ เช่น หมอ.mp4)")
            continue
        if video_duration(path) > MAX_CLIP_SEC:
            print(f"{path.name}: ข้าม — ยาวเกิน {MAX_CLIP_SEC:.0f} วินาที (1 ไฟล์ควรมี 1 คำ)")
            continue
        out = config.LANDMARK_DIR / f"{path.stem}.npz"
        if out.exists() and out.stat().st_mtime > path.stat().st_mtime:
            print(f"{path.stem:12s} (สกัดไว้แล้ว ข้าม)")
            continue
        n_frames = sum(1 for _ in iter_video(path))
        # ลองทั้งโหมดติดตาม (tracking) และโหมดภาพนิ่ง เลือกแบบที่จับมือได้มากกว่า
        best = None
        for static in (False, True):
            p, m, a, t = track_frames(path, static)
            score = m.sum() if len(m) else -1
            if best is None or score > best[0]:
                best = (score, p, m, a, t, static)
        _, p, m, a, t, static = best
        if len(p) < 4:
            print(f"{path.stem:12s} ✗ ตรวจไม่พบร่างกาย ข้ามไฟล์นี้")
            continue
        np.savez(out, points=p, mask=m, active=a, t=t, label=label_of(path.stem))
        print(f"{path.stem:12s} เฟรม {n_frames:3d} | พบร่างกาย {len(p):3d} | "
              f"มือซ้าย {int(m[:, 0].sum()):3d} มือขวา {int(m[:, 1].sum()):3d} | "
              f"ช่วงทำท่า {int(a.sum()):3d} | {'static' if static else 'tracking'}")


if __name__ == "__main__":
    main()
