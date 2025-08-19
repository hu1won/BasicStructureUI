import json, soundfile as sf
from torch.utils.data import Dataset

class ManifestDataset(Dataset):
    def __init__(self, manifest_path, processor, sample_rate=16000, with_text=True):
        self.items = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
        self.processor = processor
        self.sample_rate = sample_rate
        self.with_text = with_text

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]
        wav, sr = sf.read(x["audio"])
        if sr != self.sample_rate:
            # 간단 리샘플 (실전은 torchaudio.resample 권장)
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        sample = {"input_audio": wav, "text": x.get("text", "")}
        return sample

def collate_fn_whisper(batch, processor):
    audios = [b["input_audio"] for b in batch]
    texts  = [b["text"] for b in batch]
    inputs = processor(audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt")
    with processor.as_target_processor():
        labels = processor(texts).input_ids
    inputs["labels"] = labels
    return inputs
