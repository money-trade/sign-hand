"""ส่วน NLP: แปลงลำดับคำภาษามือ (gloss) ให้เป็นประโยคภาษาไทยตามไวยากรณ์

หลักการ: ใช้ "เฉพาะคำที่ทำท่าจริง" ไม่เติมคำใหม่ขึ้นเอง แค่จัดลำดับให้ถูกไวยากรณ์ไทย

ไวยากรณ์ภาษามือไทย (คนหูหนวก) เทียบกับภาษาไทย
  ภาษามือ : ประธาน  กรรม  กริยา  ปฏิเสธ  คำถาม      เช่น  ฉัน ยา กิน
  ภาษาไทย : ประธาน  ปฏิเสธ  กริยา  กรรม  คำถาม     เช่น  ฉันกินยา
  ตัวอย่าง  ฉัน ยา กิน ไม่    -> ฉันไม่กินยา
           ฉัน หัว ปวด       -> ฉันปวดหัว
           ฉัน โรงพยาบาล ไป  -> ฉันไปโรงพยาบาล
           หมอ ฉีดยา         -> หมอฉีดยา      (หมอ/พยาบาล ทำหน้าที่ประธานได้)
           ห้องน้ำ ที่ไหน     -> ห้องน้ำที่ไหน

ขั้นตอน
  1. Viterbi decoding  : รวมความน่าจะเป็นจากโมเดลภาพ (ML) กับแบบจำลองภาษา bigram
                         เลือกลำดับคำที่สมเหตุสมผลที่สุด (เลือกจากคำที่โมเดลเสนอเท่านั้น)
  2. ลบคำซ้ำติดกัน
  3. POS tagging       : ระบุหน้าที่คำจากพจนานุกรม
  4. แบ่งอนุประโยค + จัดลำดับใหม่เป็นไวยากรณ์ไทย
"""
import math
from collections import Counter
from dataclasses import dataclass, field

from . import config

# ---------------------------------------------------------------- พจนานุกรม
# pos: PRON ประธาน/สรรพนาม, NOUN คำนาม, VERB กริยา, NEG ปฏิเสธ, Q คำถาม, INTJ คำอุทาน/ทักทาย
# agent: คำนามที่เป็นผู้กระทำได้ (ขึ้นต้นประโยคเป็นประธาน)
LEXICON = {
    # สรรพนาม / บุคคล
    "ฉัน": {"pos": "PRON"},
    "คุณ": {"pos": "PRON"},
    "หมอ": {"pos": "NOUN", "agent": True},
    "พยาบาล": {"pos": "NOUN", "agent": True},
    # สถานที่
    "โรงพยาบาล": {"pos": "NOUN"},
    "ห้องน้ำ": {"pos": "NOUN"},
    # ยา
    "ยา": {"pos": "NOUN"},
    "ยาดม": {"pos": "NOUN"},
    "ยาทาแผล": {"pos": "NOUN"},
    # ร่างกาย / อาการ
    "หัว": {"pos": "NOUN"},
    "ท้อง": {"pos": "NOUN"},
    "ฟัน": {"pos": "NOUN"},
    "แผล": {"pos": "NOUN"},
    "เลือด": {"pos": "NOUN"},
    "ไข้": {"pos": "NOUN"},
    "น้ำ": {"pos": "NOUN"},
    # กริยา
    "กิน": {"pos": "VERB"},
    "ทา": {"pos": "VERB"},
    "ดม": {"pos": "VERB"},
    "ฉีดยา": {"pos": "VERB"},
    "ปวด": {"pos": "VERB"},
    "เจ็บ": {"pos": "VERB"},
    "ป่วย": {"pos": "VERB"},
    "แพ้": {"pos": "VERB"},
    "ไอ": {"pos": "VERB"},
    "อาเจียน": {"pos": "VERB"},
    "เวียนหัว": {"pos": "VERB"},
    "หายใจ": {"pos": "VERB"},
    "ช่วย": {"pos": "VERB"},
    "ไป": {"pos": "VERB"},
    "รอ": {"pos": "VERB"},
    "นอน": {"pos": "VERB"},
    # ปฏิเสธ / คำถาม / อื่น ๆ
    "ไม่": {"pos": "NEG"},
    "ที่ไหน": {"pos": "Q"},
    "อะไร": {"pos": "Q"},
    "เมื่อไร": {"pos": "Q"},
    "ขอบคุณ": {"pos": "INTJ"},
    "สวัสดี": {"pos": "INTJ"},
}

# คำศัพท์ที่แนะนำให้เก็บข้อมูลเพิ่มสำหรับใช้ในโรงพยาบาล (ดูท่าได้ที่ th-sl.com)
HOSPITAL_VOCAB = list(LEXICON)

POS_TH = {"PRON": "ประธาน", "VERB": "กริยา", "NOUN": "คำนาม", "NEG": "ปฏิเสธ",
          "Q": "คำถาม", "INTJ": "อุทาน"}


def pos_of(word):
    return LEXICON.get(word, {}).get("pos", "NOUN")


def is_agent(word):
    return LEXICON.get(word, {}).get("agent", False)


# ---------------------------------------------------------------- แบบจำลองภาษา
class BigramLM:
    """แบบจำลองภาษา bigram พร้อม add-k smoothing"""

    BOS, EOS = "<s>", "</s>"

    def __init__(self, sentences, vocab, k=0.1):
        self.vocab = set(vocab) | {self.EOS}
        self.k = k
        self.uni = Counter()
        self.bi = Counter()
        for s in sentences:
            toks = [self.BOS] + s + [self.EOS]
            for a, b in zip(toks, toks[1:]):
                self.uni[a] += 1
                self.bi[(a, b)] += 1

    @classmethod
    def from_file(cls, path, vocab):
        sents = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sents.append(line.split())
        return cls(sents, vocab)

    def logp(self, prev, word):
        return math.log((self.bi[(prev, word)] + self.k) /
                        (self.uni[prev] + self.k * len(self.vocab)))


@dataclass
class Composition:
    raw: list = field(default_factory=list)      # คำที่โมเดลภาพเลือก (อันดับ 1) ตามลำดับท่า
    words: list = field(default_factory=list)    # หลัง Viterbi + ลบคำซ้ำ (ยังเป็นลำดับภาษามือ)
    ordered: list = field(default_factory=list)  # ลำดับตามไวยากรณ์ไทย
    sentence: str = ""
    tagged: str = ""
    notes: list = field(default_factory=list)


class SentenceComposer:
    def __init__(self, vocab, corpus_path=config.CORPUS_PATH, lm_weight=0.6):
        self.vocab = [w for w in vocab if w != config.NONE_LABEL]
        self.lm = BigramLM.from_file(corpus_path, self.vocab)
        self.lm_weight = lm_weight

    # ---------- 1) Viterbi ----------
    def decode(self, segments):
        """segments: list ของ dict {คำ: ความน่าจะเป็น} -> ลำดับคำที่ดีที่สุด"""
        if not segments:
            return []
        V = self.vocab

        def emit(seg, w):
            total = sum(seg.get(v, 0.0) for v in V) or 1.0
            return math.log(max(seg.get(w, 0.0) / total, 1e-4))

        lam = self.lm_weight
        score = {w: emit(segments[0], w) + lam * self.lm.logp(BigramLM.BOS, w) for w in V}
        back = []
        for seg in segments[1:]:
            new, ptr = {}, {}
            for w in V:
                prev = max(V, key=lambda p: score[p] + lam * self.lm.logp(p, w))
                new[w] = score[prev] + lam * self.lm.logp(prev, w) + emit(seg, w)
                ptr[w] = prev
            score = new
            back.append(ptr)
        last = max(V, key=lambda w: score[w] + lam * self.lm.logp(w, BigramLM.EOS))
        path = [last]
        for ptr in reversed(back):
            path.append(ptr[path[-1]])
        return path[::-1]

    # ---------- 4) แบ่งอนุประโยค ----------
    @staticmethod
    def to_clauses(words):
        """แบ่งเป็นอนุประโยค แต่ละอนุประโยคเก็บคำพร้อมตำแหน่งเดิม"""
        def new():
            return {"S": None, "NEG": None, "V": None, "O": [], "Q": None, "INTJ": None}

        clauses, cur = [], new()

        def push():
            nonlocal cur
            if any(cur[k] for k in cur):
                clauses.append(cur)
            cur = new()

        for i, w in enumerate(words):
            pos = pos_of(w)
            if pos == "INTJ":
                push()
                cur["INTJ"] = (i, w)
                push()
            elif pos == "NOUN":
                # คำนามหลังจากอนุประโยคมีกริยา+กรรมแล้ว หรือหลังคำถาม = เริ่มอนุประโยคใหม่
                if cur["Q"] or (cur["V"] and cur["O"]):
                    push()
                cur["O"].append((i, w))
            else:
                slot = {"PRON": "S", "VERB": "V", "NEG": "NEG", "Q": "Q"}[pos]
                if cur[slot] is not None or (slot != "Q" and cur["Q"]):
                    push()
                cur[slot] = (i, w)
        push()
        return clauses

    @staticmethod
    def order_clause(c):
        """จัดลำดับ ประธาน-ปฏิเสธ-กริยา-กรรม-คำถาม (ใช้เฉพาะคำที่มีอยู่)"""
        if c["INTJ"]:
            return [c["INTJ"][1]]
        subj = c["S"]
        objs = list(c["O"])
        v_idx = c["V"][0] if c["V"] else None
        if v_idx is not None:
            agents_before = [o for o in objs if is_agent(o[1]) and o[0] < v_idx]
            if agents_before and (subj is None or subj[0] > v_idx):
                # หมอ ช่วย ฉัน -> หมอ(ประธาน) ช่วย ฉัน(กรรม)
                agent = agents_before[0]
                objs.remove(agent)
                if subj is not None:
                    objs.append(subj)
                subj = agent
        objs.sort(key=lambda o: o[0])
        out = [subj[1]] if subj else []
        if c["NEG"]:
            out.append(c["NEG"][1])
        if c["V"]:
            out.append(c["V"][1])
        out += [o[1] for o in objs]
        if c["Q"]:
            out.append(c["Q"][1])
        return out

    def compose(self, segments):
        """segments: list ของ dict ความน่าจะเป็นต่อท่า -> Composition"""
        res = Composition()
        if not segments:
            return res
        res.raw = [max(self.vocab, key=lambda w: s.get(w, 0.0)) for s in segments]
        decoded = self.decode(segments)
        for i, (a, b) in enumerate(zip(res.raw, decoded)):
            if a != b:
                res.notes.append(f"แบบจำลองภาษาเลือกคำที่ {i + 1} เป็น '{b}' แทน '{a}' "
                                 f"(โมเดลภาพให้คะแนนใกล้กัน)")

        words = [w for i, w in enumerate(decoded) if i == 0 or w != decoded[i - 1]]
        if len(words) < len(decoded):
            res.notes.append("ลบคำที่ซ้ำติดกัน")
        res.words = words
        res.tagged = "  ".join(f"{w}/{POS_TH.get(pos_of(w), pos_of(w))}" for w in words)

        parts = [self.order_clause(c) for c in self.to_clauses(words)]
        res.ordered = [w for p in parts for w in p]
        if res.ordered != words:
            res.notes.append("จัดลำดับจากไวยากรณ์ภาษามือเป็นไวยากรณ์ไทย (ประธาน-กริยา-กรรม)")
        # ภาษาไทยไม่เว้นวรรคระหว่างคำ เว้นวรรคเฉพาะระหว่างอนุประโยค
        res.sentence = " ".join("".join(p) for p in parts if p)
        return res
