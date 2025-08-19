import torch, json
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup
from jiwer import wer, cer
from pathlib import Path
from .base_engine import BaseEngine
from src.data.dataset import ManifestDataset, collate_fn_whisper

class WhisperEngine(BaseEngine):
    def __init__(self, cfg):
        super().__init__(cfg)
        mcfg = cfg["model"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(mcfg["name_or_path"], language=mcfg.get("lang","ko"), task=mcfg.get("task","transcribe"))
        self.model = WhisperForConditionalGeneration.from_pretrained(mcfg["name_or_path"])
        self.model.to(self.device)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=mcfg.get("lang","ko"), task=mcfg.get("task","transcribe")
        )

    def train(self):
        cfg = self.cfg
        train_ds = ManifestDataset(cfg["data"]["manifest_train"], self.processor, cfg["data"]["sample_rate"])
        dev_ds   = ManifestDataset(cfg["data"]["manifest_valid"], self.processor, cfg["data"]["sample_rate"])

        collate = lambda b: collate_fn_whisper(b, self.processor)
        dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=collate)
        optim = torch.optim.AdamW(self.model.parameters(), lr=cfg["train"]["lr"])
        total_steps = cfg["train"]["max_steps"]
        sched = get_linear_schedule_with_warmup(optim, int(total_steps*0.1), total_steps)
        self.model.train()

        step, log_every, save_every = 0, cfg["train"]["log_every"], cfg["train"]["save_every"]
        best_dev = 1e9
        out_dir = Path(cfg["output"]["ckpt_dir"]).with_name(cfg["exp_name"])
        ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)

        while step < total_steps:
            for batch in dl:
                step += 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                loss.backward()
                optim.step(); sched.step(); self.model.zero_grad()

                if step % log_every == 0:
                    print(f"[{step}/{total_steps}] loss={loss.item():.4f}")

                if step % save_every == 0 or step == total_steps:
                    dev_metrics = self.evaluate(self.cfg["data"]["manifest_valid"])
                    if dev_metrics["wer"] < best_dev:
                        best_dev = dev_metrics["wer"]
                        torch.save(self.model.state_dict(), ckpt_dir / "best.pt")
                    torch.save(self.model.state_dict(), ckpt_dir / f"step{step}.pt")

                if step >= total_steps: break

        self._dump_metrics({"best_dev_wer": best_dev}, "dev_metrics.json")

    @torch.no_grad()
    def evaluate(self, manifest_path: str, save_to: str | None = None) -> dict:
        self.model.eval()
        items = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
        refs, hyps = [], []
        for it in items:
            hyp = self.infer_file(it["audio"])
            refs.append(it.get("text",""))
            hyps.append(hyp)
        m = {"wer": wer(refs, hyps), "cer": cer(refs, hyps)}
        if save_to:
            with open(save_to, "w", encoding="utf-8") as f: json.dump(m, f, ensure_ascii=False, indent=2)
        print("[EVAL]", m)
        return m

    @torch.no_grad()
    def infer_file(self, wav_path: str) -> str:
        import soundfile as sf, numpy as np, librosa
        wav, sr = sf.read(wav_path)
        if sr != self.processor.feature_extractor.sampling_rate:
            wav = librosa.resample(wav, sr, self.processor.feature_extractor.sampling_rate)
        inputs = self.processor(np.array(wav), sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt").to(self.device)
        gen = self.model.generate(**inputs, num_beams=self.cfg["decode"].get("beam_size",5))
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return text.strip()
