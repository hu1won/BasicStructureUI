import torch, json
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, get_linear_schedule_with_warmup
from jiwer import wer, cer
from pathlib import Path
from typing import Optional, Dict
from .base_engine import BaseEngine
from src.data.dataset import ManifestDataset

class Wav2Vec2Engine(BaseEngine):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        m = cfg["model"]["name_or_path"]
        self.processor = Wav2Vec2Processor.from_pretrained(m)
        self.model = Wav2Vec2ForCTC.from_pretrained(m)
        self.model.to(self.device)

    def _collate(self, batch):
        audios = [b["input_audio"] for b in batch]
        texts  = [b["text"] for b in batch]
        inputs = self.processor(audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        with self.processor.as_target_processor():
            labels = self.processor(texts, return_tensors="pt", padding=True).input_ids
        inputs["labels"] = labels
        return inputs

    def train(self):
        cfg = self.cfg
        train_ds = ManifestDataset(cfg["data"]["manifest_train"], self.processor, cfg["data"]["sample_rate"])
        dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=self._collate)
        optim = torch.optim.AdamW(self.model.parameters(), lr=cfg["train"]["lr"])
        total = cfg["train"]["max_steps"]; warm = int(total*0.1)
        sched = get_linear_schedule_with_warmup(optim, warm, total)
        self.model.train()

        step, log_every = 0, cfg["train"]["log_every"]
        out_dir = Path(cfg["output"]["ckpt_dir"]).with_name(cfg["exp_name"])
        ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
        best = 1e9

        while step < total:
            for batch in dl:
                step += 1
                batch = {k: v.to(self.device) for k,v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                loss.backward()
                optim.step(); sched.step(); self.model.zero_grad()

                if step % log_every == 0:
                    print(f"[{step}/{total}] loss={loss.item():.4f}")

                if step % cfg["train"]["save_every"] == 0 or step == total:
                    dev_metrics = self.evaluate(cfg["data"]["manifest_valid"])
                    if dev_metrics["wer"] < best:
                        best = dev_metrics["wer"]
                        torch.save(self.model.state_dict(), ckpt_dir / "best.pt")

                if step >= total: break

        self._dump_metrics({"best_dev_wer": best}, "dev_metrics.json")

    @torch.no_grad()
    def evaluate(self, manifest_path: str, save_to: Optional[str] = None) -> Dict:
        self.model.eval()
        items = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
        refs, hyps = [], []
        for it in items:
            hyp = self.infer_file(it["audio"])
            refs.append(it.get("text","")); hyps.append(hyp)
        m = {"wer": wer(refs, hyps), "cer": cer(refs, hyps)}
        if save_to:
            with open(save_to, "w", encoding="utf-8") as f: json.dump(m, f, ensure_ascii=False, indent=2)
        print("[EVAL]", m)
        return m

    @torch.no_grad()
    def infer_file(self, wav_path: str) -> str:
        import soundfile as sf, numpy as np, librosa
        wav, sr = sf.read(wav_path)
        tgt_sr = self.processor.feature_extractor.sampling_rate
        if sr != tgt_sr: wav = librosa.resample(wav, sr, tgt_sr)
        inputs = self.processor(np.array(wav), sampling_rate=tgt_sr, return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(ids)[0]
        return text.strip()
