#!/usr/bin/env python3
"""
실제 파일명에 맞춰 올바른 매니페스트 생성
"""

import os
import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ipa.to_ipa import KoreanToIPA

def create_correct_manifest():
    """실제 파일명에 맞춰 올바른 매니페스트를 생성합니다"""
    
    # IPA 변환기 초기화
    ipa_converter = KoreanToIPA()
    
    # 실제 오디오 파일 목록
    audio_dir = "data/raw/wav"
    audio_files = []
    
    if os.path.exists(audio_dir):
        for file in os.listdir(audio_dir):
            if file.endswith('.wav'):
                audio_files.append(file)
    
    audio_files.sort()
    print(f"📁 발견된 오디오 파일: {len(audio_files)}개")
    
    # 매니페스트 데이터 생성
    manifest_data = []
    
    for i, audio_file in enumerate(audio_files, 1):
        # 파일명에서 단어 추출 (word_XX_단어.wav → 단어)
        parts = audio_file.replace('.wav', '').split('_')
        if len(parts) >= 3:
            word = '_'.join(parts[2:])  # word_01_바지 → 바지
        else:
            continue
        
        # IPA 변환
        ipa = ipa_converter.text_to_ipa(word)
        
        # 매니페스트 아이템 생성
        item = {
            "audio_file": f"data/raw/wav/{audio_file}",
            "transcript_file": f"data/raw/transcripts/{audio_file.replace('.wav', '.txt')}",
            "expected_text": word,
            "actual_pronunciation": word,  # 기본값 (필요시 수동 수정)
            "actual_ipa": ipa,
            "expected_ipa": ipa,
            "word_id": i,
            "word": word,
            "duration": None,
            "category": "pronunciation_based",
            "difficulty": "easy"
        }
        
        manifest_data.append(item)
        print(f"  {i:2d}. {audio_file} → {word} → {ipa}")
    
    # 훈련/검증 분할 (80:20)
    total_samples = len(manifest_data)
    train_samples = int(total_samples * 0.8)
    
    train_data = manifest_data[:train_samples]
    val_data = manifest_data[train_samples:]
    
    print(f"\n📊 데이터 분할:")
    print(f"  훈련 샘플: {len(train_data)}개")
    print(f"  검증 샘플: {len(val_data)}개")
    
    # 매니페스트 파일 저장
    output_dir = "data"
    
    # 전체 매니페스트
    full_manifest_path = f"{output_dir}/manifest_correct_word_30.json"
    with open(full_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)
    
    # 훈련 매니페스트
    train_manifest_path = f"{output_dir}/manifest_correct_word_30_train.json"
    with open(train_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 검증 매니페스트
    val_manifest_path = f"{output_dir}/manifest_correct_word_30_val.json"
    with open(val_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 올바른 매니페스트 생성 완료:")
    print(f"  - 전체 매니페스트: {full_manifest_path}")
    print(f"  - 훈련 매니페스트: {train_manifest_path}")
    print(f"  - 검증 매니페스트: {val_manifest_path}")

if __name__ == "__main__":
    create_correct_manifest() 