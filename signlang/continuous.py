"""จับท่าภาษามือแบบต่อเนื่องจากสตรีมกล้อง

การทำงาน
  1. ยกมือเหนือเส้น = เริ่มช่วงทำท่า (session) เก็บภาพ landmark พร้อมเวลา
  2. ระหว่างทำท่า ทุก ๆ ANALYZE_EVERY (2 วิ) วิเคราะห์ภาพที่เก็บไว้ทันที
     ไม่ต้องรอลดมือ -> ทำท่าหลายคำติดกันได้
  3. ลดมือลง = จบ session วิเคราะห์ส่วนที่เหลือทั้งหมด

การหาคำในช่วงภาพ (detect_signs)
  - Sliding window หลายความยาว (0.5-2.6 วิ) เลื่อนทีละ 0.1 วิ ทำนายทุกหน้าต่างพร้อมกัน (batch)
  - โมเดลถูกฝึกให้ตอบ _none เมื่อหน้าต่างมีท่าไม่ครบหรือคร่อมสองท่า
  - Non-Maximum Suppression เลือกหน้าต่างที่มั่นใจที่สุดซึ่งไม่ซ้อนทับกัน
  - เรียงตามเวลา -> ได้ลำดับคำ
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Detection:
    word: str
    conf: float
    probs: dict
    t0: float
    t1: float


def make_windows(ts):
    """สร้างหน้าต่าง (i0, i1) ตามเวลาจริงของแต่ละเฟรม"""
    dur = ts[-1] - ts[0]
    wins = set()
    for L in config.WINDOW_SECONDS:
        if L > dur + 0.05:
            continue
        for s in np.arange(ts[0], ts[-1] - L + 1e-6, config.WINDOW_HOP):
            i0 = int(np.searchsorted(ts, s))
            i1 = int(np.searchsorted(ts, s + L, side="right"))
            if i1 - i0 >= 5:
                wins.add((i0, i1))
    if dur >= config.MIN_SIGN_SEC and dur <= max(config.WINDOW_SECONDS) and len(ts) >= 5:
        wins.add((0, len(ts)))   # ทั้งช่วง (กรณีทำท่าเดียวแล้วลดมือ)
    return sorted(wins)


def detect_signs(recognizer, points, mask, ts, threshold=config.CONF_THRESHOLD):
    """หาลำดับคำในช่วงภาพ -> list ของ Detection เรียงตามเวลา"""
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < 5 or ts[-1] - ts[0] < config.MIN_SIGN_SEC:
        return []
    wins = make_windows(ts)
    if not wins:
        return []
    probs = recognizer.predict_batch([(points[a:b], mask[a:b]) for a, b in wins])
    labels = recognizer.labels
    none_i = labels.index(config.NONE_LABEL) if config.NONE_LABEL in labels else -1

    cands = []
    for (a, b), p in zip(wins, probs):
        q = p.copy()
        if none_i >= 0:
            q[none_i] = -1
        w = int(q.argmax())
        if p[w] >= threshold and (none_i < 0 or p[w] > p[none_i]):
            t0, t1 = ts[a], ts[b - 1]
            # ความมั่นใจเท่ากัน -> เลือกหน้าต่างที่ยาวกว่า (ครอบคลุมท่าเต็มกว่า)
            cands.append((p[w] + 0.05 * (t1 - t0), w, p, t0, t1))

    cands.sort(key=lambda c: -c[0])
    chosen = []
    for score, w, p, t0, t1 in cands:
        ok = True
        for c in chosen:
            inter = min(t1, c.t1) - max(t0, c.t0)
            if inter > config.NMS_OVERLAP * min(t1 - t0, c.t1 - c.t0):
                ok = False
                break
        if ok:
            chosen.append(Detection(labels[w], float(p[w]), dict(zip(labels, p.tolist())), t0, t1))
    chosen.sort(key=lambda d: d.t0)

    # คำเดียวกันติดกันแทบไม่มีช่องว่าง = ท่าเดียวที่ถูกนับซ้ำ
    merged = []
    for d in chosen:
        if merged and merged[-1].word == d.word and d.t0 - merged[-1].t1 < 0.25:
            if d.conf > merged[-1].conf:
                d.t0 = merged[-1].t0
                merged[-1] = d
            else:
                merged[-1].t1 = d.t1
        else:
            merged.append(d)
    return merged


class StreamRecognizer:
    """รับเฟรมทีละเฟรม คืนค่า event:
        ("words", [Detection, ...])   คำที่ยืนยันแล้ว
        ("analyzed", n)                เพิ่งวิเคราะห์เสร็จ (n = จำนวนคำที่ได้)
        ("sample", points, mask, t)    ช่วงทำท่า 1 ครั้ง (โหมดเก็บข้อมูล)
    """

    def __init__(self, recognizer=None):
        self.rec = recognizer
        self.threshold = config.CONF_THRESHOLD
        self.collect = False
        self.now = 0.0
        self.reset()

    def reset(self):
        self.p, self.m, self.t = [], [], []
        self.sess_p, self.sess_m, self.sess_t = [], [], []
        self.in_session = False
        self.active_run = 0
        self.last_active = 0.0
        self.last_analysis = 0.0
        self.pre = deque(maxlen=60)     # ภาพก่อนเริ่ม (pre-roll)
        self.active_start = 0.0

    @property
    def cycle_progress(self):
        """ความคืบหน้าจนถึงการวิเคราะห์รอบถัดไป (0-1)"""
        if not self.in_session:
            return 0.0
        return min((self.now - self.last_analysis) / config.ANALYZE_EVERY, 1.0)

    def update(self, frame, active, t):
        """frame = (points, mask) หรือ None, active = ยกมือเหนือเส้นหรือไม่, t = เวลา (วินาที)"""
        self.now = t
        if frame is not None:
            self.pre.append((frame[0], frame[1], t))

        if not self.in_session:
            if active and self.active_run == 0:
                self.active_start = t
            self.active_run = self.active_run + 1 if active else 0
            if self.active_run >= config.SEG_START_FRAMES and self.pre:
                self.in_session = True
                self.last_active = t
                # ใส่ภาพช่วงยกมือขึ้นก่อนเริ่มด้วย (ท่าในข้อมูลฝึกเริ่มจากจังหวะยกมือ)
                for p, m, ft in self.pre:
                    if ft >= self.active_start - config.PRE_ROLL_SEC:
                        self._append(p, m, ft)
                self.last_analysis = self.t[0]
            return []

        if frame is not None:
            self._append(frame[0], frame[1], t)
        if active:
            self.last_active = t

        if t - self.last_active >= config.SESSION_END_SEC:
            return self._finish()
        if not self.collect and t - self.last_analysis >= config.ANALYZE_EVERY:
            return self._analyze(final=False)
        return []

    def flush(self):
        """จบ session ทันที (เช่น สตรีมวิดีโอจบ)"""
        return self._finish() if self.in_session else []

    # ------------------------------------------------------------
    def _append(self, p, m, t):
        self.p.append(p)
        self.m.append(m)
        self.t.append(t)
        self.sess_p.append(p)
        self.sess_m.append(m)
        self.sess_t.append(t)

    def _finish(self):
        events = []
        if self.collect:
            T = np.array(self.sess_t)
            keep = T <= self.last_active + config.POST_ROLL_SEC
            if keep.sum() >= 5 and T[keep][-1] - T[0] >= config.MIN_SIGN_SEC:
                events.append(("sample", np.array(self.sess_p)[keep],
                               np.array(self.sess_m)[keep], T[keep]))
        elif self.p:
            events = self._analyze(final=True)
        self.reset()
        return events

    def _analyze(self, final):
        self.last_analysis = self.now
        if self.rec is None or not self.p:
            return []
        P, M, T = np.array(self.p), np.array(self.m), np.array(self.t)
        if final:
            keep = T <= self.last_active + config.POST_ROLL_SEC   # ตัดเฟรมตอนลดมือลงทิ้ง
            P, M, T = P[keep], M[keep], T[keep]
        dets = detect_signs(self.rec, P, M, T, self.threshold)

        if final:
            commit = dets
        else:
            # ท่าที่จบใกล้ขอบท้ายอาจยังทำไม่เสร็จ -> รอวิเคราะห์รวมกับภาพรอบถัดไป
            commit = [d for d in dets if d.t1 <= self.now - config.TAIL_MARGIN]
            if commit:
                cut = max(d.t1 for d in commit) - 0.1
            else:
                cut = self.now - (config.MAX_BUFFER_SEC - config.ANALYZE_EVERY)
            start = int(np.searchsorted(T, cut))
            self.p, self.m, self.t = self.p[start:], self.m[start:], self.t[start:]

        events = [("analyzed", len(commit))]
        if commit:
            events.append(("words", commit))
        return events
