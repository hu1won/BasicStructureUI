#!/usr/bin/env python3
"""
학습된 간단한 CNN 모델로 실제 음성 인식 테스트
"""

import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engines.simple_cnn_engine import SimpleCNNEngine
from src.utils.io import load_config
from src.data.ipa_dataset import IPADataset


def load_audio_file(audio_path: str, target_length: int = 48000, sample_rate: int = 16000):
    """오디오 파일 로드 및 전처리"""
    print(f"오디오 파일 로드: {audio_path}")
    
    # 오디오 로드
    waveform, sr = torchaudio.load(audio_path)
    
    # 샘플 레이트 변환 (필요한 경우)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # 모노로 변환
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    print(f"  원본 shape: {waveform.shape}, sample_rate: {sample_rate}")
    
    # 길이 조정
    current_length = waveform.shape[1]
    if current_length < target_length:
        # 패딩
        padding = int(target_length - current_length)
        waveform = torch.nn.functional.pad(waveform, (0, padding), mode='replicate')
        print(f"  길이 패딩 후: {waveform.shape}")
    elif current_length > target_length:
        # 자르기
        target_length_int = int(target_length)
        waveform = waveform[:, :target_length_int]
        print(f"  길이 자르기 후: {waveform.shape}")
    
    return waveform


def extract_mfcc_features(waveform: torch.Tensor, config: dict):
    """MFCC 특징 추출"""
    print(f"MFCC 추출 시작: waveform shape = {waveform.shape}")
    
    # MFCC 변환기 생성
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=config['data']['sample_rate'],
        n_mfcc=config['data']['audio']['mfcc']['n_mfcc'],
        melkwargs={
            'n_mels': 80,
            'n_fft': config['data']['audio']['mfcc']['n_fft'],
            'hop_length': config['data']['audio']['mfcc']['hop_length']
        }
    )
    
    # MFCC 추출
    mfcc_features = mfcc_transform(waveform)
    print(f"MFCC 변환 후: {mfcc_features.shape}")
    
    # 차원 조정
    if mfcc_features.dim() == 3:
        mfcc_features = mfcc_features.squeeze(0)
    print(f"MFCC squeeze 후: {mfcc_features.shape}")
    
    print(f"MFCC 최종 결과: {mfcc_features.shape}")
    return mfcc_features


def test_single_audio(audio_path: str, engine: SimpleCNNEngine, config: dict):
    """단일 오디오 파일 테스트"""
    print(f"\n{'='*50}")
    print(f"음성 인식 테스트: {os.path.basename(audio_path)}")
    print(f"{'='*50}")
    
    try:
        # 1. 오디오 로드 및 전처리
        waveform = load_audio_file(audio_path, config['data']['max_duration'] * config['data']['sample_rate'])
        
        # 2. MFCC 특징 추출
        mfcc_features = extract_mfcc_features(waveform, config)
        
        # 3. 예측
        predictions = engine.predict(mfcc_features.unsqueeze(0))
        
        # 4. 결과 출력
        print(f"\n🎯 예측 결과:")
        for i, pred in enumerate(predictions):
            print(f"  샘플 {i+1}: {pred}")
        
        return predictions[0] if predictions else "예측 실패"
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return f"오류: {e}"


def test_multiple_audio(audio_dir: str, engine: SimpleCNNEngine, config: dict):
    """여러 오디오 파일 테스트"""
    print(f"\n{'='*60}")
    print(f"여러 음성 파일 테스트: {audio_dir}")
    print(f"{'='*60}")
    
    # WAV 파일 찾기
    wav_files = []
    for ext in ['*.wav', '*.WAV']:
        wav_files.extend(Path(audio_dir).glob(ext))
    
    if not wav_files:
        print(f"❌ WAV 파일을 찾을 수 없습니다: {audio_dir}")
        return
    
    print(f"📁 발견된 WAV 파일: {len(wav_files)}개")
    
    # 각 파일 테스트
    results = {}
    for wav_file in sorted(wav_files):
        result = test_single_audio(str(wav_file), engine, config)
        results[wav_file.name] = result
    
    # 결과 요약
    print(f"\n{'='*60}")
    print(f"테스트 결과 요약")
    print(f"{'='*60}")
    
    for filename, result in results.items():
        status = "✅" if not result.startswith("오류") else "❌"
        print(f"{status} {filename}: {result}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="학습된 모델로 음성 인식 테스트")
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    parser.add_argument('--model_path', type=str, required=True, help='학습된 모델 경로')
    parser.add_argument('--audio_path', type=str, required=True, help='테스트할 오디오 파일 또는 디렉토리')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    print(f"설정 파일 로드 완료: {args.config}")
    
    # 엔진 초기화
    engine = SimpleCNNEngine(config)
    print("엔진 초기화 완료")
    
    # 모델 로드
    engine.load_model(args.model_path)
    print(f"모델 로드 완료: {args.model_path}")
    
    # 테스트 실행
    if os.path.isfile(args.audio_path):
        # 단일 파일 테스트
        test_single_audio(args.audio_path, engine, config)
    elif os.path.isdir(args.audio_path):
        # 디렉토리 테스트
        test_multiple_audio(args.audio_path, engine, config)
    else:
        print(f"❌ 파일 또는 디렉토리를 찾을 수 없습니다: {args.audio_path}")


if __name__ == "__main__":
    main() 