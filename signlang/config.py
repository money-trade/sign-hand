"""ค่าตั้งค่ากลางของทั้งโปรเจกต์"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# โฟลเดอร์ชุดข้อมูลวิดีโอ/GIF (ชื่อไฟล์ = คำศัพท์ เช่น กิน.gif, หมอ_2.mp4)
GIF_DIR = BASE_DIR / "dataset" if (BASE_DIR / "dataset").exists() else BASE_DIR
VIDEO_EXTS = (".gif", ".mp4", ".avi", ".mov", ".mkv", ".webm")

DATA_DIR = BASE_DIR / "data"
LANDMARK_DIR = DATA_DIR / "landmarks"   # landmark ที่สกัดจากวิดีโอ/GIF
CUSTOM_DIR = DATA_DIR / "custom"        # ตัวอย่างที่ผู้ใช้บันทึกเพิ่มจากกล้อง
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sign_model.pt"
CORPUS_PATH = Path(__file__).resolve().parent / "corpus.txt"

NONE_LABEL = "_none"    # คลาส "ไม่ใช่ท่าภาษามือ / ท่าไม่ครบ / คาบเกี่ยวสองท่า"

# ---- การแปลงลำดับท่าทาง ----
SEQ_LEN = 32            # ทุกท่าถูก resample ให้เหลือ 32 เฟรมก่อนเข้าโมเดล
BASE_FPS = 25           # ข้อมูลฝึกทุกชุดถูกปรับให้เป็นอัตราเฟรมเดียวกัน

# ท่าถือว่า "กำลังทำ" เมื่อข้อมือสูงกว่าระดับนี้
# (หน่วย = ความกว้างไหล่ วัดลงจากระดับไหล่, ค่าน้อย = สูง)
ACTIVE_Y = 1.25

# ---- การจับท่าแบบต่อเนื่อง (เรียลไทม์) ----
SEG_START_FRAMES = 3    # ต้องยกมือต่อเนื่องกี่เฟรมจึงเริ่มจับ
PRE_ROLL_SEC = 0.3      # เก็บภาพก่อนยกมือพ้นเส้นด้วย (จังหวะเริ่มยกมือ)
POST_ROLL_SEC = 0.25    # เก็บภาพหลังลดมือลงเล็กน้อย
SESSION_END_SEC = 0.5   # ลดมือลงนานเท่านี้ = จบช่วงทำท่า -> วิเคราะห์ทันที
ANALYZE_EVERY = 2.0     # ระหว่างทำท่าต่อเนื่อง จับภาพแล้ววิเคราะห์ทุก ๆ 2 วินาที
TAIL_MARGIN = 0.8       # ท่าที่จบภายใน 0.8 วิสุดท้าย รอรอบถัดไป (อาจยังทำไม่จบ)
MAX_BUFFER_SEC = 6.0    # เก็บภาพในหน่วยความจำไม่เกินนี้
MIN_SIGN_SEC = 0.35     # ช่วงที่สั้นกว่านี้ถือว่าเป็นสัญญาณรบกวน

# sliding window สำหรับหาคำในช่วงที่ทำท่าต่อเนื่อง
WINDOW_SECONDS = (0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2.1, 2.6)
WINDOW_HOP = 0.1        # เลื่อนหน้าต่างทีละ 0.1 วินาที
NMS_OVERLAP = 0.3       # ท่าที่เลือกซ้อนทับกันได้ไม่เกิน 30%

CONF_THRESHOLD = 0.55   # ความมั่นใจขั้นต่ำที่จะยอมรับคำ
SENTENCE_TIMEOUT = 3.5  # หยุดทำท่ากี่วินาทีจึงสรุปเป็นประโยค

TH_SL_SEARCH = "https://www.th-sl.com/?s={}"   # ดูท่าตัวอย่างจากฐานข้อมูลภาษามือไทย
