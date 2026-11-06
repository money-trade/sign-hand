"""ขั้นที่ 2: เทรนโมเดลจำแนกท่าภาษามือ

    python train.py [--per-class 500] [--epochs 25]

ใช้ข้อมูลจาก data/landmarks (สกัดจากวิดีโอ/GIF) และ data/custom/<คำ>/*.npz
(ตัวอย่างที่บันทึกเพิ่มจากกล้องในโหมดเก็บข้อมูลของ app.py)
"""
import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from signlang import config
from signlang.augment import (augment, partial_sign, straddle, synth_random_motion,
                              with_context, wrist_reference)
from signlang.features import FEAT_DIM, to_rate, trim_active
from extract_landmarks import is_word_name
from signlang.model import SignGRU


def load_sources():
    """คืนค่า {คำ: [(points, mask), ...]} ทุกชุดปรับเป็น 25 fps ตามเวลาจริง"""
    sources = {}
    for f in sorted(config.LANDMARK_DIR.glob("*.npz")):
        d = np.load(f)
        if "label" not in d or not is_word_name(str(d["label"])):
            print(f"ข้าม {f.name}: ไม่ใช่ไฟล์คำศัพท์")
            continue
        if "t" not in d:
            raise SystemExit(f"{f.name} เป็นรูปแบบเก่า — ลบโฟลเดอร์ data/landmarks แล้วรัน extract_landmarks.py ใหม่")
        sl = trim_active(d["active"])
        p, m = to_rate(d["points"][sl], d["mask"][sl], d["t"][sl])
        sources.setdefault(str(d["label"]), []).append((p, m))
    if config.CUSTOM_DIR.exists():
        for label_dir in sorted(config.CUSTOM_DIR.iterdir()):
            for f in sorted(label_dir.glob("*.npz")):
                d = np.load(f)
                t = d["t"] if "t" in d else np.arange(len(d["points"])) / 30.0
                if len(t) < 4:
                    continue
                sources.setdefault(label_dir.name, []).append(to_rate(d["points"], d["mask"], t))
    if not sources:
        raise SystemExit("ไม่พบข้อมูล landmark — รัน python extract_landmarks.py ก่อน")
    return sources


def build_dataset(sources, labels, per_class, rng):
    """ตัวอย่างบวก per_class ต่อคำ + ตัวอย่าง _none 1.5 เท่า
    (ท่าไม่ครบ 40%, คร่อมสองท่า 40%, มือขยับมั่ว 20%)"""
    X, y = [], []
    refs = []
    all_src = [s for v in sources.values() for s in v]
    for li, label in enumerate(labels):
        if label == config.NONE_LABEL:
            continue
        samples = sources[label]
        others = [s for l, v in sources.items() if l != label for s in v]
        refs.extend(wrist_reference(p, m) for p, m in samples)
        for k in range(per_class):
            p, m = with_context(*samples[k % len(samples)], others, rng)
            X.append(augment(p, m, rng))
            y.append(li)

    none_idx = labels.index(config.NONE_LABEL)
    n_none = int(per_class * 1.5)
    while len(y) < (len(labels) - 1) * per_class + n_none:
        r = rng.random()
        if r < 0.4:
            x = augment(*partial_sign(*all_src[rng.integers(len(all_src))], rng), rng, crop=0)
        elif r < 0.8:
            a, b = (all_src[i] for i in rng.integers(len(all_src), size=2))
            x = augment(*straddle(a, b, rng), rng, crop=0)
        else:
            x = synth_random_motion(all_src, rng, refs=refs)
        if x is not None:
            X.append(x)
            y.append(none_idx)
    return np.stack(X), np.array(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sources = load_sources()
    labels = sorted(sources) + [config.NONE_LABEL]
    print("คำศัพท์:", ", ".join(labels))
    for k, v in sources.items():
        print(f"  {k:10s} ตัวอย่างต้นแบบ {len(v)}")

    # ชุด validation สร้างจาก seed อื่น (augmentation ไม่ซ้ำกับชุดเทรน)
    Xtr, ytr = build_dataset(sources, labels, args.per_class, np.random.default_rng(args.seed))
    Xva, yva = build_dataset(sources, labels, max(60, args.per_class // 5),
                             np.random.default_rng(args.seed + 1000))
    mean = Xtr.reshape(-1, FEAT_DIM).mean(0)
    std = Xtr.reshape(-1, FEAT_DIM).std(0) + 1e-6
    norm = lambda X: torch.from_numpy((X - mean) / std).float()

    train_dl = DataLoader(TensorDataset(norm(Xtr), torch.from_numpy(ytr)),
                          batch_size=64, shuffle=True)
    Xva_t, yva_t = norm(Xva), torch.from_numpy(yva)

    model = SignGRU(FEAT_DIM, len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, epochs=args.epochs,
                                                steps_per_epoch=len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(1)
        acc = (pred == yva_t).float().mean().item()
        if acc >= best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d} | loss {total / len(train_dl.dataset):.4f} | val acc {acc:.3f}")

    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(Xva_t).argmax(1).numpy()
    print("\nConfusion matrix (แถว = คำจริง, คอลัมน์ = คำที่ทำนาย)")
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(yva, pred):
        cm[t, p] += 1
    for i, l in enumerate(labels):
        print(f"  {l:10s}", " ".join(f"{v:4d}" for v in cm[i]))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "labels": labels, "seq_len": config.SEQ_LEN,
        "in_dim": FEAT_DIM, "mean": mean.astype(np.float32), "std": std.astype(np.float32),
    }, config.MODEL_PATH)
    print(f"\nบันทึกโมเดลที่ {config.MODEL_PATH} (val acc {best_acc:.3f})")


if __name__ == "__main__":
    main()
