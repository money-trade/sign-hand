"""ทดสอบทั้งระบบแบบเดียวกับตอนใช้งานจริง (ไม่ต้องใช้กล้อง)

    python evaluate.py                       # ทดสอบด้วย GIF ในชุดข้อมูล
    python evaluate.py --video rec.mp4       # ทดสอบกับวิดีโอที่อัดไว้
        [--crop x0,y0,x1,y1] [--mirror] [--start 0] [--end 15]

ป้อนเฟรมทีละเฟรมผ่าน MediaPipe -> จับท่าต่อเนื่อง (sliding window + NMS) -> โมเดล -> NLP
"""
import argparse

import cv2
import numpy as np

from extract_landmarks import iter_video
from signlang import config
from signlang.continuous import StreamRecognizer
from signlang.features import is_active, trim_active
from signlang.model import SignRecognizer
from signlang.nlp import SentenceComposer
from signlang.tracker import HolisticTracker

# ลำดับท่าตามไวยากรณ์ภาษามือ (ประธาน กรรม กริยา) -> ข้อความที่ควรได้
SENTENCES = [
    (["ฉัน", "ยา", "กิน"], "ฉันกินยา"),
    (["ฉัน", "กิน", "ยา"], "ฉันกินยา"),
    (["ยา", "ฉัน", "กิน"], "ฉันกินยา"),
    (["ฉัน", "ยาทาแผล", "ทา"], "ฉันทายาทาแผล"),
    (["ฉัน", "ยาดม"], "ฉันยาดม"),
    (["ฉัน", "ยา", "กิน", "ยาทาแผล", "ทา"], "ฉันกินยา ทายาทาแผล"),
]

GRAMMAR = [
    "ฉัน ยา กิน", "ฉัน ยา กิน ไม่", "ฉัน หัว ปวด", "ฉัน ท้อง ปวด ยา กิน", "ฉัน โรงพยาบาล ไป",
    "หมอ ฉีดยา", "หมอ ช่วย ฉัน", "ห้องน้ำ ที่ไหน", "ฉัน ยา แพ้", "ยาดม", "ขอบคุณ หมอ",
]


def run_stream(clips, rec, rest_between=True):
    """clips: list ของ list (rgb, dt) — ป้อนต่อกันเป็นสตรีมเดียว คืนค่า Detection ทั้งหมด"""
    stream = StreamRecognizer(rec)
    dets, t = [], 0.0
    with HolisticTracker(model_complexity=1) as tracker:
        def feed(rgb, dt, force_idle=False):
            nonlocal t
            _, fr = tracker.process(rgb.copy())
            active = (not force_idle) and fr is not None and is_active(*fr)
            for ev in stream.update(fr[:2] if fr is not None else None, active, t):
                if ev[0] == "words":
                    dets.extend(ev[1])
            t += dt

        for clip in clips:
            for rgb, dt in clip:
                feed(rgb, dt)
            if rest_between:   # พักมือ (ลดมือลง) ระหว่างคำ
                for _ in range(int(config.SESSION_END_SEC / 0.04) + 3):
                    feed(clip[-1][0], 0.04, force_idle=True)
        for ev in stream.flush():
            if ev[0] == "words":
                dets.extend(ev[1])
    return dets


def load_clip(path, active_only=False):
    frames = list(iter_video(path))
    clip = [(f, (frames[i + 1][1] - t) if i + 1 < len(frames) else 0.04)
            for i, (f, t) in enumerate(frames)]
    if active_only:
        d = np.load(config.LANDMARK_DIR / f"{path.stem}.npz")
        if len(d["active"]) == len(clip):
            clip = clip[trim_active(d["active"], pad=0)]
    return clip


def fmt(dets):
    return " ".join(f"{d.word}({d.conf * 100:.0f}%)" for d in dets) or "(ไม่พบท่า)"


def eval_gifs(rec, composer):
    # ทดสอบเฉพาะคำที่โมเดลรู้จัก (GIF ที่เพิ่มใหม่แต่ยังไม่ได้เทรนจะถูกข้าม)
    paths = {p.stem: p for p in sorted(config.GIF_DIR.glob("*.gif")) if p.stem in rec.words}
    clips = {w: load_clip(p) for w, p in paths.items()}
    active = {w: load_clip(p, active_only=True) for w, p in paths.items()}

    print("== 1) ทีละคำ ==")
    ok_n = total = 0
    for mirror in (False, True):
        for w, clip in clips.items():
            c = [(f[:, ::-1].copy(), dt) for f, dt in clip] if mirror else clip
            dets = run_stream([c], rec)
            ok = [d.word for d in dets] == [w]
            ok_n += ok
            total += 1
            print(f"  {'[กลับด้าน] ' if mirror else ''}{w:8s} -> {fmt(dets)} {'✓' if ok else '✗'}")
    print(f"  ถูก {ok_n}/{total}\n")

    for title, rest in (("2) ประโยค — ลดมือลงระหว่างคำ", True),
                        ("3) ประโยค — ทำท่าต่อเนื่องไม่ลดมือ", False)):
        print(f"== {title} ==")
        ok_n = 0
        for words, expect in SENTENCES:
            src = clips if rest else active
            dets = run_stream([src[w] for w in words], rec, rest_between=rest)
            res = composer.compose([d.probs for d in dets])
            ok = res.sentence == expect
            ok_n += ok
            print(f"  ท่า {' '.join(words):28s} ตรวจได้ {fmt(dets)}")
            print(f"     -> ข้อความ: {res.sentence or '-'}  {'✓' if ok else '✗ (ควรได้ ' + expect + ')'}")
        print(f"  ถูก {ok_n}/{len(SENTENCES)}\n")


def eval_grammar(composer):
    print("== 4) ไวยากรณ์ (ป้อนลำดับท่าโดยตรง) ==")
    vocab_comp = SentenceComposer(list({w for g in GRAMMAR for w in g.split()}))
    for g in GRAMMAR:
        res = vocab_comp.compose([{w: 1.0} for w in g.split()])
        print(f"  {g:24s} -> {res.sentence}")
    amb = [{"ฉัน": 0.95}, {"ยา": 0.9}, {"ทา": 0.45, "กิน": 0.40}]
    r = composer.compose(amb)
    print(f"  ฉัน ยา [ทา45%/กิน40%]      -> {r.sentence}   ({'; '.join(r.notes)})")


def eval_video(rec, composer, args):
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    crop = [int(v) for v in args.crop.split(",")] if args.crop else None
    clip = []
    while True:
        ok, bgr = cap.read()
        if not ok or (args.end and len(clip) / fps + args.start > args.end):
            break
        if crop:
            bgr = bgr[crop[1]:crop[3], crop[0]:crop[2]]
        if args.mirror:
            bgr = bgr[:, ::-1]
        clip.append((cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB), 1 / fps))
    dets = run_stream([clip], rec, rest_between=False)
    print(f"== วิดีโอ {args.video} ({args.start:.0f}-{args.start + len(clip) / fps:.0f} วิ) ==")
    for d in dets:
        print(f"  {args.start + d.t0:5.1f}-{args.start + d.t1:5.1f} วิ  {d.word:8s} {d.conf * 100:.0f}%")
    res = composer.compose([d.probs for d in dets])
    print(f"  ลำดับท่า: {' '.join(res.words) or '-'}")
    print(f"  ข้อความ : {res.sentence or '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--crop", help="x0,y0,x1,y1")
    ap.add_argument("--mirror", action="store_true", help="วิดีโอถูกกลับด้านแบบกระจก")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=0)
    args = ap.parse_args()

    rec = SignRecognizer()
    composer = SentenceComposer(rec.words)
    if args.video:
        eval_video(rec, composer, args)
    else:
        eval_gifs(rec, composer)
        eval_grammar(composer)


if __name__ == "__main__":
    main()
