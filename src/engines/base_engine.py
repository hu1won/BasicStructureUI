from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Union
import json, time

class BaseEngine(ABC):
    def __init__(self, config):
        self.config = config
        # exp_dir 설정은 선택적으로 (필요한 경우에만)
        if "output" in config and "ckpt_dir" in config["output"] and "exp_name" in config:
            self.exp_dir = Path(config["output"]["ckpt_dir"]).parent / config["exp_name"]
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.exp_dir = None

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def evaluate(self, manifest_path: str, save_to: Optional[str] = None) -> Dict:
        ...

    @abstractmethod
    def infer_file(self, wav_path: str) -> str:
        ...

    def _dump_metrics(self, metrics: Dict, filename: str = "metrics.json"):
        out = self.exp_dir / filename
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return out

    def _tic(self): self._t0 = time.time()
    def _toc(self): return (time.time() - self._t0)
