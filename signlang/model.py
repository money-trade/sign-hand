"""โมเดลจำแนกท่าภาษามือ (Bidirectional GRU) และตัวช่วยโหลด/ทำนาย"""
import numpy as np
import torch
from torch import nn

from . import config
from .features import prepare_sequence


class SignGRU(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=128):
        super().__init__()
        # ใส่ทั้งตำแหน่งและความเร็ว (ผลต่างระหว่างเฟรม) เข้าโมเดล
        self.inp = nn.Sequential(
            nn.Linear(in_dim * 2, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.2),
        )
        self.gru = nn.GRU(192, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        vel = torch.diff(x, dim=1, prepend=x[:, :1])
        h = self.inp(torch.cat([x, vel * 4.0], dim=-1))
        out, _ = self.gru(h)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=-1)
        return self.head(pooled)


class SignRecognizer:
    """โหลดโมเดลที่เทรนแล้ว และทำนายคำจากลำดับ landmark"""

    def __init__(self, path=config.MODEL_PATH):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.labels = ckpt["labels"]
        self.seq_len = ckpt["seq_len"]
        self.mean = ckpt["mean"]
        self.std = ckpt["std"]
        self.model = SignGRU(ckpt["in_dim"], len(self.labels))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @property
    def words(self):
        return [l for l in self.labels if l != config.NONE_LABEL]

    def predict_batch(self, segments):
        """segments: list ของ (points, mask) -> array ความน่าจะเป็น (N, จำนวนคลาส)"""
        X = np.stack([prepare_sequence(np.asarray(p), np.asarray(m), self.seq_len)
                      for p, m in segments])
        X = (X - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float())
            return torch.softmax(logits, dim=-1).numpy()

    def predict(self, points, mask):
        """คืนค่า dict {คำ: ความน่าจะเป็น}"""
        probs = self.predict_batch([(points, mask)])[0]
        return dict(zip(self.labels, probs.tolist()))
