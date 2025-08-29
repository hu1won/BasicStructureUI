#!/usr/bin/env python3
"""
모델 디버깅 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchaudio
from src.engines.simple_cnn_engine import SimpleCNNEngine
from src.utils.io import load_config

def debug_model_output():
    """모델 출력 디버깅"""
    
    # 설정 로드
    config = load_config('configs/simple_cnn_training.yaml')
    
    # 엔진 초기화
    engine = SimpleCNNEngine(config)
    
    # 모델 로드
    engine.load_model('outputs/simple_cnn_training/final_model.pth')
    
    # 테스트 오디오 로드
    audio_path = 'data/raw/wav/word_01_바지.wav'
    waveform, sr = torchaudio.load(audio_path)
    
    # 모노로 변환
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 길이 조정 (3초)
    target_length = 48000
    if waveform.shape[1] < target_length:
        padding = int(target_length - waveform.shape[1])
        waveform = torch.nn.functional.pad(waveform, (0, padding), mode='replicate')
    else:
        waveform = waveform[:, :target_length]
    
    # MFCC 추출
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={'n_mels': 80, 'n_fft': 2048, 'hop_length': 64}
    )
    
    mfcc_features = mfcc_transform(waveform)
    if mfcc_features.dim() == 3:
        mfcc_features = mfcc_features.squeeze(0)
    
    print(f"MFCC shape: {mfcc_features.shape}")
    
    # 모델 예측
    engine.model.eval()
    with torch.no_grad():
        # raw 출력 확인
        outputs = engine.model(mfcc_features.unsqueeze(0))
        logits = outputs['logits']
        
        print(f"Logits shape: {logits.shape}")
        print(f"Logits sample: {logits[0, :10]}")  # 첫 10개 값
        
        # 확률 분포 확인
        probs = torch.softmax(logits, dim=-1)
        print(f"Probabilities sum: {torch.sum(probs, dim=-1)}")
        print(f"Max probability: {torch.max(probs)}")
        print(f"Min probability: {torch.min(probs)}")
        
        # top-5 예측
        top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
        print(f"\nTop-5 predictions:")
        for i in range(5):
            idx = top5_indices[0, i].item()
            prob = top5_probs[0, i].item()
            token = engine.vocab[idx] if idx < len(engine.vocab) else f"<{idx}>"
            print(f"  {i+1}. {token} (prob: {prob:.4f}, idx: {idx})")

if __name__ == "__main__":
    debug_model_output() 