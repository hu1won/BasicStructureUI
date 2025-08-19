import argparse, yaml
from src.engines.whisper_engine import WhisperEngine
from src.engines.wav2vec2_engine import Wav2Vec2Engine
ENGINES = {"whisper": WhisperEngine, "wav2vec2": Wav2Vec2Engine}

def load_cfg(p):
    cfg = yaml.safe_load(open(p))
    if "inherit" in cfg:
        base = yaml.safe_load(open(f"configs/{cfg['inherit']}")); base.update({k:v for k,v in cfg.items() if k!="inherit"}); cfg=base
    return cfg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    eng = ENGINES[cfg["engine"]](cfg)
    print(eng.infer_file(args.audio))
