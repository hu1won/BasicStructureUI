#!/usr/bin/env python3
"""
학습 데이터 검증 스크립트
"""

import os
import sys
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.ipa_dataset import IPADataset
from src.utils.io import load_config

def validate_manifest(manifest_path: str):
    """매니페스트 파일 검증"""
    print(f"=== 매니페스트 검증: {manifest_path} ===")
    
    if not os.path.exists(manifest_path):
        print(f"❌ 매니페스트 파일이 존재하지 않습니다: {manifest_path}")
        return False
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 매니페스트 파일 로드 성공")
        print(f"📊 총 샘플 수: {len(data)}")
        
        # 첫 번째 샘플 구조 확인
        if data:
            first_sample = data[0]
            print(f"🔍 첫 번째 샘플 구조:")
            for key, value in first_sample.items():
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 매니페스트 파일 로드 실패: {e}")
        return False

def validate_audio_files(manifest_path: str):
    """오디오 파일 존재 여부 검증"""
    print(f"\n=== 오디오 파일 검증 ===")
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        missing_files = []
        existing_files = []
        
        for item in data:
            audio_path = item.get('audio_file', '')  # audio_path 대신 audio_file 사용
            if audio_path and os.path.exists(audio_path):
                existing_files.append(audio_path)
            else:
                missing_files.append(audio_path)
        
        print(f"✅ 존재하는 오디오 파일: {len(existing_files)}개")
        print(f"❌ 누락된 오디오 파일: {len(missing_files)}개")
        
        if missing_files:
            print(f"🔍 누락된 파일들:")
            for file in missing_files[:5]:  # 처음 5개만 표시
                print(f"  {file}")
            if len(missing_files) > 5:
                print(f"  ... 외 {len(missing_files) - 5}개")
        
        return len(missing_files) == 0
        
    except Exception as e:
        print(f"❌ 오디오 파일 검증 실패: {e}")
        return False

def validate_dataset_creation(config_path: str):
    """데이터셋 생성 테스트"""
    print(f"\n=== 데이터셋 생성 테스트 ===")
    
    try:
        config = load_config(config_path)
        print(f"✅ 설정 파일 로드 성공")
        
        # 훈련 데이터셋 생성 테스트
        train_manifest = config['data']['train_manifest']
        if os.path.exists(train_manifest):
            train_dataset = IPADataset(
                manifest_path=train_manifest,
                config=config,
                is_training=True
            )
            print(f"✅ 훈련 데이터셋 생성 성공: {len(train_dataset)}개 샘플")
            print(f"🔍 어휘 크기: {len(train_dataset.vocab)}")
            
            # 첫 번째 샘플 테스트
            if len(train_dataset) > 0:
                first_item = train_dataset[0]
                print(f"🔍 첫 번째 샘플:")
                for key, value in first_item.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {value}")
            
            return True
        else:
            print(f"❌ 훈련 매니페스트 파일이 존재하지 않습니다: {train_manifest}")
            return False
            
    except Exception as e:
        print(f"❌ 데이터셋 생성 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("=== 학습 데이터 검증 시작 ===\n")
    
    # 1. 매니페스트 파일 검증
    train_manifest = "data/manifest_correct_word_30_train.json"
    val_manifest = "data/manifest_correct_word_30_val.json"
    
    train_manifest_ok = validate_manifest(train_manifest)
    val_manifest_ok = validate_manifest(val_manifest)
    
    # 2. 오디오 파일 검증
    audio_ok = False
    if train_manifest_ok:
        audio_ok = validate_audio_files(train_manifest)
    
    # 3. 데이터셋 생성 테스트
    dataset_ok = False
    if train_manifest_ok and audio_ok:
        dataset_ok = validate_dataset_creation("configs/simple_cnn_training.yaml")
    
    # 4. 최종 결과 요약
    print(f"\n=== 검증 결과 요약 ===")
    print(f"📋 훈련 매니페스트: {'✅' if train_manifest_ok else '❌'}")
    print(f"📋 검증 매니페스트: {'✅' if val_manifest_ok else '❌'}")
    print(f"🎵 오디오 파일: {'✅' if audio_ok else '❌'}")
    print(f"📊 데이터셋 생성: {'✅' if dataset_ok else '❌'}")
    
    if all([train_manifest_ok, val_manifest_ok, audio_ok, dataset_ok]):
        print(f"\n🎉 모든 검증 통과! 학습을 진행할 수 있습니다.")
    else:
        print(f"\n⚠️ 일부 검증 실패. 문제를 해결한 후 학습을 진행하세요.")

if __name__ == "__main__":
    main() 