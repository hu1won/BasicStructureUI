from abc import ABC, abstractmethod
from pathlib import Path
import json, time

class BaseEngine(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.exp_dir = Path(cfg["output"]["ckpt_dir"]).parent / cfg["exp_name"]
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def evaluate(self, manifest_path: str, save_to: str | None = None) -> dict:
        ...

    @abstractmethod
    def infer_file(self, wav_path: str) -> str:
        ...

    def _dump_metrics(self, metrics: dict, filename: str = "metrics.json"):
        out = self.exp_dir / filename
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return out

    def _tic(self): self._t0 = time.time()
    def _toc(self): return (time.time() - self._t0)
