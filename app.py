"""ขั้นที่ 3: โปรแกรมแปลภาษามือไทยเป็นข้อความแบบเรียลไทม์

    python app.py                 # ใช้กล้องตัวที่ 0
    python app.py --camera 1      # เลือกกล้องอื่น (เลือกในหน้าจอได้เช่นกัน)
    python app.py --source x.mp4  # ทดสอบกับไฟล์วิดีโอ/GIF

หน้าจอ: ซ้าย = ภาพกล้องพร้อมโครงร่างที่ตรวจจับได้, ขวา = คำที่ตรวจจับและประโยคที่เรียบเรียง
วิธีใช้: ยกมือขึ้นเหนือเส้นประแล้วทำท่าต่อเนื่องได้หลายคำ ระบบวิเคราะห์ทุก 2 วินาที
         ลดมือลง = วิเคราะห์ทันที, หยุดทำท่าสักครู่ = สรุปเป็นประโยค
"""
import argparse
import queue
import threading
import time
import webbrowser
from datetime import datetime
from urllib.parse import quote

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

from signlang import config
from signlang.cameras import list_cameras
from signlang.continuous import StreamRecognizer
from signlang.features import is_active
from signlang.nlp import HOSPITAL_VOCAB, SentenceComposer
from signlang.tracker import HolisticTracker, draw_landmarks

FONT = "Leelawadee UI"
BG = "#f4f6fb"
CARD = "#ffffff"
ACCENT = "#2f6fed"
REC = "#e5484d"
MUTED = "#6b7280"
CAM_W, CAM_H = 800, 600


class CameraWorker(threading.Thread):
    """อ่านกล้อง + MediaPipe + จับท่าต่อเนื่อง + โมเดล ในเธรดแยก (หน้าจอไม่ค้าง)"""

    def __init__(self, source, recognizer, threshold, collect):
        super().__init__(daemon=True)
        self.source = source
        self.stream = StreamRecognizer(recognizer)
        self.stream.threshold = threshold
        self.stream.collect = collect
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.status = {"session": False, "progress": 0.0, "fps": 0.0, "person": False}
        self.running = True
        self.error = None

    def _emit(self, events):
        for e in events:
            self.events.put(e)

    def run(self):
        is_cam = isinstance(self.source, int)
        if is_cam:
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            fps_file = 0
        else:
            cap = cv2.VideoCapture(self.source)
            fps_file = cap.get(cv2.CAP_PROP_FPS) or 25
        if not cap.isOpened():
            self.error = f"เปิดกล้อง/ไฟล์ไม่ได้: {self.source}"
            return

        t_prev = time.time()
        fails = 0
        n_frame = 0
        with HolisticTracker(model_complexity=1) as tracker:
            while self.running:
                t0 = time.time()
                ok, bgr = cap.read()
                if not ok:
                    if is_cam:
                        fails += 1
                        if fails > 60:
                            self.error = "อ่านภาพจากกล้องไม่ได้\n(กล้องอาจถูกโปรแกรมอื่นใช้อยู่ ลองเลือกกล้องอื่น)"
                            break
                        time.sleep(0.03)
                        continue
                    self._emit(self.stream.flush())   # วิดีโอจบ: วิเคราะห์ส่วนที่เหลือ
                    break
                fails = 0
                # เวลาของเฟรม: กล้อง = เวลาจริง, ไฟล์ = ตามอัตราเฟรมของไฟล์
                t = time.time() if is_cam else n_frame / fps_file
                n_frame += 1

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results, frame = tracker.process(rgb)
                active = frame is not None and is_active(*frame)
                self._emit(self.stream.update(frame[:2] if frame is not None else None, active, t))

                draw_landmarks(bgr, results)
                self._draw_threshold(bgr, results)
                view = cv2.flip(bgr, 1)   # แสดงแบบกระจกให้ผู้ใช้ดูง่าย
                if self.stream.in_session:
                    cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (77, 72, 229), 10)
                    cv2.circle(view, (40, 40), 12, (77, 72, 229), -1)
                    cv2.putText(view, "REC", (60, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (77, 72, 229), 3)

                now = time.time()
                fps = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                self.status = {"session": self.stream.in_session,
                               "progress": self.stream.cycle_progress,
                               "fps": 0.9 * self.status["fps"] + 0.1 * fps,
                               "person": frame is not None}
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))

                if fps_file:
                    time.sleep(max(0.0, 1.0 / fps_file - (time.time() - t0)))
        cap.release()
        self.status = {**self.status, "session": False, "progress": 0.0}

    @staticmethod
    def _draw_threshold(bgr, results):
        """เส้นประบอกระดับที่ต้องยกมือให้สูงกว่า จึงจะเริ่มจับท่า"""
        if results.pose_landmarks is None:
            return
        h, w = bgr.shape[:2]
        lm = results.pose_landmarks.landmark
        ls = np.array([lm[11].x * w, lm[11].y * h])
        rs = np.array([lm[12].x * w, lm[12].y * h])
        y = int((ls[1] + rs[1]) / 2 + config.ACTIVE_Y * np.linalg.norm(ls - rs))
        if 0 < y < h:
            for x in range(0, w, 30):
                cv2.line(bgr, (x, y), (x + 15, y), (0, 220, 120), 2)


class App(tk.Tk):
    def __init__(self, source):
        super().__init__()
        self.title("แปลภาษามือไทยเป็นข้อความ (Thai Sign Language → Text)")
        self.configure(bg=BG)
        self.geometry("1300x760")
        self.minsize(1200, 720)

        self.recognizer = None
        self.composer = None
        self.load_error = None
        try:
            from signlang.model import SignRecognizer
            self.recognizer = SignRecognizer()
            self.composer = SentenceComposer(self.recognizer.words)
        except Exception as e:  # ยังไม่ได้เทรนโมเดล -> ใช้ได้เฉพาะโหมดเก็บข้อมูล
            self.load_error = str(e)

        self.segments = []          # ความน่าจะเป็นของแต่ละคำในประโยคปัจจุบัน (เรียงตามเวลา)
        self.last_sign_time = None
        self.threshold = tk.DoubleVar(value=config.CONF_THRESHOLD)
        self.collect = tk.BooleanVar(value=self.recognizer is None)
        self.collect_label = tk.StringVar()

        self.cameras = []           # [(index, ชื่ออุปกรณ์)]
        self.camera_var = tk.StringVar()
        self.worker = None

        self._build_ui()
        self.refresh_cameras(select=source if isinstance(source, int) else None)
        self.start_source(source)
        self.bind("<space>", lambda e: self.finalize())
        self.bind("<BackSpace>", lambda e: self.undo())
        self.bind("<Escape>", lambda e: self.clear())
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        if self.load_error:
            self.notes_var.set("⚠ ยังไม่พบโมเดล: รัน extract_landmarks.py และ train.py ก่อน\n"
                               "(ตอนนี้ใช้ได้เฉพาะโหมดเก็บข้อมูล)")
        self.after(15, self.loop)

    # ------------------------------------------------------------ UI
    def _card(self, parent, title):
        frame = tk.Frame(parent, bg=CARD, highlightbackground="#e3e7ef", highlightthickness=1)
        frame.pack(fill="x", pady=(0, 8))
        tk.Label(frame, text=title, bg=CARD, fg=MUTED, font=(FONT, 10, "bold"),
                 anchor="w").pack(fill="x", padx=12, pady=(6, 0))
        return frame

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Rec.Horizontal.TProgressbar", troughcolor="#e8ebf2", background=REC)
        style.configure("TButton", font=(FONT, 10), padding=5)

        left_col = tk.Frame(self, bg=BG)
        left_col.pack(side="left", padx=12, pady=12, anchor="n")
        left = tk.Frame(left_col, bg="#111", width=CAM_W, height=CAM_H)
        left.pack()
        left.pack_propagate(False)
        self.cam_label = tk.Label(left, bg="#111", fg="#ddd", font=(FONT, 14),
                                  text="กำลังเปิดกล้อง...")
        self.cam_label.pack(expand=True, fill="both")

        # เลือกอุปกรณ์กล้อง / ไฟล์วิดีโอ
        dev = tk.Frame(left_col, bg=BG)
        dev.pack(fill="x", pady=(8, 0))
        tk.Label(dev, text="กล้อง (Input)", bg=BG, font=(FONT, 10, "bold")).pack(side="left")
        self.camera_box = ttk.Combobox(dev, textvariable=self.camera_var, state="readonly",
                                       width=42, font=(FONT, 10))
        self.camera_box.pack(side="left", padx=6)
        self.camera_box.bind("<<ComboboxSelected>>", lambda e: self.on_camera_selected())
        ttk.Button(dev, text="รีเฟรชรายการ", command=self.refresh_cameras).pack(side="left")
        ttk.Button(dev, text="เปิดไฟล์วิดีโอ...", command=self.open_video).pack(side="left", padx=6)

        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)

        tk.Label(right, text="แปลภาษามือ → ข้อความ", bg=BG, fg="#111827",
                 font=(FONT, 18, "bold"), anchor="w").pack(fill="x")

        # สถานะ
        st = tk.Frame(right, bg=BG)
        st.pack(fill="x", pady=(2, 6))
        self.dot = tk.Canvas(st, width=14, height=14, bg=BG, highlightthickness=0)
        self.dot_id = self.dot.create_oval(2, 2, 12, 12, fill=MUTED, outline="")
        self.dot.pack(side="left")
        self.status_var = tk.StringVar(value="รอท่าทาง")
        tk.Label(st, textvariable=self.status_var, bg=BG, fg="#374151",
                 font=(FONT, 11)).pack(side="left", padx=6)
        self.fps_var = tk.StringVar()
        tk.Label(st, textvariable=self.fps_var, bg=BG, fg=MUTED, font=(FONT, 9)).pack(side="right")
        self.progress = ttk.Progressbar(right, style="Rec.Horizontal.TProgressbar", maximum=1.0)
        self.progress.pack(fill="x", pady=(0, 8))

        # คำล่าสุด
        c = self._card(right, "คำที่ตรวจจับล่าสุด")
        row = tk.Frame(c, bg=CARD)
        row.pack(fill="x", padx=12, pady=(0, 6))
        self.word_var = tk.StringVar(value="—")
        tk.Label(row, textvariable=self.word_var, bg=CARD, fg=ACCENT, wraplength=330,
                 justify="left", font=(FONT, 20, "bold")).pack(side="left")
        self.top3_var = tk.StringVar()
        tk.Label(row, textvariable=self.top3_var, bg=CARD, fg=MUTED, justify="right",
                 font=(FONT, 9)).pack(side="right")

        # ลำดับคำ
        c = self._card(right, "ลำดับท่าภาษามือ (gloss)")
        self.gloss_var = tk.StringVar(value="ยกมือขึ้นเหนือเส้นประเพื่อเริ่มทำท่า")
        tk.Label(c, textvariable=self.gloss_var, bg=CARD, fg="#111827", font=(FONT, 13),
                 wraplength=440, justify="left", anchor="w").pack(fill="x", padx=12, pady=(2, 8))

        # ประโยค
        c = self._card(right, "ข้อความ (เรียงตามไวยากรณ์ไทย)")
        self.sentence_var = tk.StringVar(value="")
        tk.Label(c, textvariable=self.sentence_var, bg=CARD, fg="#0f172a", font=(FONT, 24, "bold"),
                 wraplength=440, justify="left", anchor="w").pack(fill="x", padx=12)
        self.notes_var = tk.StringVar()
        tk.Label(c, textvariable=self.notes_var, bg=CARD, fg=MUTED, font=(FONT, 9),
                 wraplength=440, justify="left", anchor="w").pack(fill="x", padx=12, pady=(2, 8))
        self.countdown_var = tk.StringVar()
        tk.Label(right, textvariable=self.countdown_var, bg=BG, fg=ACCENT,
                 font=(FONT, 10)).pack(fill="x")

        # ปุ่มควบคุม
        ctl = tk.Frame(right, bg=BG)
        ctl.pack(fill="x", pady=(4, 4))
        ttk.Button(ctl, text="สรุปประโยคทันที (Space)", command=self.finalize).pack(side="left")
        ttk.Button(ctl, text="ลบคำล่าสุด", command=self.undo).pack(side="left", padx=6)
        ttk.Button(ctl, text="ล้าง (Esc)", command=self.clear).pack(side="left")

        opt = tk.Frame(right, bg=BG)
        opt.pack(fill="x", pady=(2, 2))
        tk.Label(opt, text="ความมั่นใจขั้นต่ำ", bg=BG, font=(FONT, 10)).pack(side="left")
        tk.Scale(opt, variable=self.threshold, from_=0.2, to=0.95, resolution=0.05,
                 orient="horizontal", length=160, bg=BG, highlightthickness=0,
                 font=(FONT, 8)).pack(side="left", padx=4)

        # โหมดเก็บข้อมูล (บันทึกท่าของผู้ใช้เป็นข้อมูลฝึกเพิ่ม)
        col = tk.Frame(right, bg=BG)
        col.pack(fill="x", pady=(2, 0))
        tk.Checkbutton(col, text="โหมดเก็บข้อมูล บันทึกท่าเป็นคำ:", variable=self.collect,
                       bg=BG, font=(FONT, 10)).pack(side="left")
        known = self.recognizer.words if self.recognizer else []
        words = known + [w for w in HOSPITAL_VOCAB if w not in known]
        self.collect_label.set(words[0])
        box = ttk.Combobox(col, textvariable=self.collect_label, values=words, width=12)
        box.pack(side="left", padx=4)
        box.bind("<<ComboboxSelected>>", lambda e: self.update_collect_count())
        col2 = tk.Frame(right, bg=BG)
        col2.pack(fill="x", pady=(2, 6))
        ttk.Button(col2, text="ดูท่าตัวอย่าง (th-sl.com)", command=self.open_reference).pack(side="left")
        self.collect_var = tk.StringVar()
        tk.Label(col2, textvariable=self.collect_var, bg=BG, fg=MUTED, font=(FONT, 9),
                 wraplength=260, justify="left").pack(side="left", padx=6)
        self.update_collect_count()

        tk.Label(right, text="ประวัติข้อความ", bg=BG, fg=MUTED, font=(FONT, 10, "bold"),
                 anchor="w").pack(fill="x")
        self.history = tk.Listbox(right, font=(FONT, 12), height=4, bd=0,
                                  highlightthickness=1, highlightbackground="#e3e7ef")
        self.history.pack(fill="both", expand=True)

    # ------------------------------------------------------------ loop
    def loop(self):
        w = self.worker
        w.stream.threshold = self.threshold.get()
        w.stream.collect = self.collect.get()
        if w.error:
            self.cam_label.configure(text=w.error, image="")
        try:
            self._show_frame(w.frames.get_nowait())
        except queue.Empty:
            pass

        while True:
            try:
                ev = w.events.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "words":
                self.on_words(ev[1])
            elif ev[0] == "analyzed" and ev[1] == 0 and not w.stream.in_session:
                self.word_var.set("ไม่พบท่าที่รู้จัก")
            elif ev[0] == "sample":
                self.save_sample(*ev[1:])

        s = w.status
        self.dot.itemconfigure(self.dot_id, fill=REC if s["session"] else
                               ("#22c55e" if s["person"] else MUTED))
        if s["session"]:
            mode = "กำลังบันทึกตัวอย่าง..." if self.collect.get() else "กำลังจับท่าทาง..."
            self.status_var.set(mode)
        else:
            self.status_var.set("พร้อม — ยกมือขึ้นเพื่อทำท่า" if s["person"] else "ไม่พบผู้ใช้ในกล้อง")
        self.progress["value"] = s["progress"]
        self.fps_var.set(f"{s['fps']:.0f} FPS")

        if s["session"] and not self.collect.get():
            remain = config.ANALYZE_EVERY * (1 - s["progress"])
            self.countdown_var.set(f"กำลังจับภาพ — วิเคราะห์ในอีก {remain:.1f} วินาที "
                                   f"(ทำท่าต่อเนื่องได้เลย)")
        elif self.segments and self.last_sign_time:
            remain = config.SENTENCE_TIMEOUT - (time.time() - self.last_sign_time)
            if remain <= 0:
                self.finalize()
            else:
                self.countdown_var.set(f"สรุปข้อความในอีก {remain:.1f} วินาที (ทำท่าต่อเพื่อเพิ่มคำ)")
        self.after(15, self.loop)

    def _show_frame(self, rgb):
        h, w = rgb.shape[:2]
        scale = min(CAM_W / w, CAM_H / h)
        img = Image.fromarray(rgb).resize((int(w * scale), int(h * scale)))
        self._photo = ImageTk.PhotoImage(img)
        self.cam_label.configure(image=self._photo, text="")

    # ------------------------------------------------------------ logic
    def on_words(self, dets):
        """คำใหม่จากการวิเคราะห์ 1 รอบ (อาจมีหลายคำถ้าทำท่าต่อเนื่อง)"""
        self.word_var.set("  ".join(f"{d.word} {d.conf * 100:.0f}%" for d in dets))
        ranked = sorted(dets[-1].probs.items(), key=lambda kv: -kv[1])
        self.top3_var.set("\n".join(f"{w if w != config.NONE_LABEL else '(ไม่ใช่ท่า)'} "
                                    f"{p * 100:.0f}%" for w, p in ranked[:3]))
        self.segments += [d.probs for d in dets]
        self.last_sign_time = time.time()
        self.update_preview()

    def update_preview(self):
        if not self.segments:
            self.gloss_var.set("ยกมือขึ้นเหนือเส้นประเพื่อเริ่มทำท่า")
            self.notes_var.set("")
            return None
        res = self.composer.compose(self.segments)
        self.gloss_var.set(" → ".join(res.words) + f"\n{res.tagged}")
        self.sentence_var.set(res.sentence)
        self.notes_var.set("\n".join(res.notes))
        return res

    def finalize(self):
        if not self.segments or not self.composer:
            return
        res = self.update_preview()
        if res and res.sentence:
            self.history.insert(0, f"{datetime.now():%H:%M:%S}  {res.sentence}"
                                   f"   ({' '.join(res.words)})")
        self.segments = []
        self.last_sign_time = None
        self.countdown_var.set("✓ สรุปข้อความแล้ว — ทำท่าใหม่เพื่อเริ่มประโยคถัดไป")

    def undo(self):
        if self.segments:
            self.segments.pop()
            self.last_sign_time = time.time()
            self.update_preview()
            if not self.segments:
                self.sentence_var.set("")

    def clear(self):
        self.segments = []
        self.last_sign_time = None
        self.word_var.set("—")
        self.top3_var.set("")
        self.sentence_var.set("")
        self.countdown_var.set("")
        self.update_preview()

    # ------------------------------------------------------------ data collection
    def update_collect_count(self):
        label = self.collect_label.get().strip()
        d = config.CUSTOM_DIR / label
        n = len(list(d.glob("*.npz"))) if label and d.exists() else 0
        trained = self.recognizer and label in self.recognizer.words
        self.collect_var.set(f"'{label}': บันทึกแล้ว {n} ครั้ง "
                             f"{'(มีในโมเดลแล้ว)' if trained else '(ยังไม่มีในโมเดล)'} "
                             f"— ควรบันทึก 10-20 ครั้ง แล้วรัน run.bat train")

    def open_reference(self):
        """เปิดหน้าค้นหาคำในฐานข้อมูลภาษามือไทย ให้ผู้ใช้ดูท่าแล้วทำตามหน้ากล้อง"""
        label = self.collect_label.get().strip()
        if label:
            webbrowser.open(config.TH_SL_SEARCH.format(quote(label)))

    def save_sample(self, points, mask, t):
        label = self.collect_label.get().strip()
        if not label:
            self.collect_var.set("เลือก/พิมพ์คำก่อน")
            return
        d = config.CUSTOM_DIR / label
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{datetime.now():%Y%m%d_%H%M%S_%f}.npz",
                 points=points, mask=mask, t=t - t[0])
        self.word_var.set(f"บันทึก: {label}")
        self.update_collect_count()

    # ------------------------------------------------------------ input device
    def refresh_cameras(self, select=None):
        """ค้นหากล้องที่ต่ออยู่ใหม่ แล้วอัปเดตรายการให้เลือก"""
        self.cameras = list_cameras()
        items = [f"{i}: {name}" for i, name in self.cameras]
        self.camera_box["values"] = items
        if not items:
            self.camera_var.set("ไม่พบกล้อง")
            return
        if select is None and isinstance(getattr(self.worker, "source", None), int):
            select = self.worker.source
        idx = [i for i, _ in self.cameras]
        if select in idx:
            self.camera_var.set(items[idx.index(select)])

    def on_camera_selected(self):
        pos = self.camera_box.current()
        if 0 <= pos < len(self.cameras):
            index = self.cameras[pos][0]
            if not (self.worker and self.worker.source == index and self.worker.is_alive()):
                self.start_source(index)

    def open_video(self):
        path = filedialog.askopenfilename(
            title="เลือกไฟล์วิดีโอภาษามือ",
            filetypes=[("วิดีโอ/GIF", "*.mp4 *.avi *.mov *.mkv *.webm *.gif"), ("ทุกไฟล์", "*.*")])
        if path:
            self.start_source(path)

    def start_source(self, source):
        """หยุดกล้องเดิม แล้วเริ่มอ่านจากกล้อง (index) หรือไฟล์วิดีโอ (path) ใหม่"""
        self.stop_worker()
        if isinstance(source, str):
            self.camera_var.set(f"ไฟล์: {source.replace(chr(92), '/').split('/')[-1]}")
        self._photo = None
        self.cam_label.configure(image="", text="กำลังเปิดกล้อง...")
        self.worker = CameraWorker(source, self.recognizer, self.threshold.get(), self.collect.get())
        self.worker.start()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.running = False
            self.worker.join(timeout=3)
            self.worker = None

    def on_close(self):
        if self.worker is not None:
            self.worker.running = False
        self.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="ไฟล์วิดีโอ/GIF แทนกล้อง")
    args = ap.parse_args()
    App(args.source if args.source else args.camera).mainloop()


if __name__ == "__main__":
    main()
