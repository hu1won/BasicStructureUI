#!/usr/bin/env python3
"""
실제 발음 기반 매니페스트 생성 스크립트
음성의 실제 발음을 IPA로 변환하고, 이를 텍스트로 변환하는 방식
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트를 Python 경로에 추가
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.ipa.to_ipa import KoreanToIPA
from src.utils.io import save_json, ensure_dir

def create_pronunciation_manifest():
    """실제 발음이 포함된 30개 단어 매니페스트를 생성합니다."""
    
    # 30개 단어와 실제 발음 매핑
    word_pronunciations = [
        ("바지", "바지"),
        ("가방", "가방"),
        ("접시", "접찌"),
        ("장갑", "장갑"),
        ("뽀뽀", "뽀뽀"),
        ("포크", "포크"),
        ("단추", "단추"),
        ("침대", "침대"),
        ("숟가락", "숟까락"),
        ("꽃", "꼳"),
        ("딸기", "딸기"),
        ("목도리", "목또리"),
        ("토끼", "토끼"),
        ("코", "코"),
        ("사탕", "사탕"),
        ("우산", "우산"),
        ("눈사람", "눈싸람"),
        ("휴지", "휴지"),
        ("비행기", "비행기"),
        ("라면", "라면"),
        ("나무", "나무"),
        ("그네", "그네"),
        ("양말", "양말"),
        ("머리", "머리"),
        ("나비", "나비"),
        ("아프다", "아프다"), 
        ("싸우다", "싸우다"),  
        ("짹짹", "짹짹"),   
        ("먹다", "먹따"), 
        ("웃다", "웃따")
    ]
    
    # IPA 변환기 초기화
    ipa_converter = KoreanToIPA()
    
    # 매니페스트 데이터 생성
    manifest_data = []
    
    for i, (expected_text, actual_pronunciation) in enumerate(word_pronunciations, 1):
        # 오디오 파일 경로
        audio_file = f"data/raw/wav/word_{i:02d}_{expected_text}.wav"
        
        # 전사본 파일 경로
        transcript_file = f"data/raw/transcripts/word_{i:02d}_{expected_text}.txt"
        
        # 실제 발음을 IPA로 변환
        try:
            actual_ipa = ipa_converter.text_to_ipa(actual_pronunciation)
        except Exception as e:
            print(f"IPA 변환 오류 ({actual_pronunciation}): {e}")
            actual_ipa = actual_pronunciation  # 오류 시 원본 사용
        
        # 기대하는 텍스트를 IPA로 변환 (참조용)
        try:
            expected_ipa = ipa_converter.text_to_ipa(expected_text)
        except Exception as e:
            print(f"IPA 변환 오류 ({expected_text}): {e}")
            expected_ipa = expected_text  # 오류 시 원본 사용
        
        # 매니페스트 항목 생성
        manifest_item = {
            'audio_file': audio_file,
            'transcript_file': transcript_file,
            'expected_text': expected_text,      # 기대하는 텍스트 (예: "바지")
            'actual_pronunciation': actual_pronunciation,  # 실제 발음 (예: "아지")
            'actual_ipa': actual_ipa,           # 실제 발음의 IPA (예: "ɐdʑi")
            'expected_ipa': expected_ipa,       # 기대하는 IPA (예: "pɐdʑi")
            'word_id': i,
            'word': expected_text,
            'duration': None,  # 나중에 계산
            'category': 'pronunciation_based',
            'difficulty': 'easy' if expected_text == actual_pronunciation else 'hard'
        }
        
        manifest_data.append(manifest_item)
    
    # 훈련/검증 분할 (8:2)
    train_data = manifest_data[:24]  # 24개 단어
    val_data = manifest_data[24:]    # 6개 단어
    
    # 전체 매니페스트
    full_manifest = {
        'train': train_data,
        'val': val_data,
        'total_samples': len(manifest_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'split_ratio': 0.8,
        'target_words': [word for word, _ in word_pronunciations],
        'training_approach': 'pronunciation_based_ipa',
        'description': '음성의 실제 발음을 IPA로 변환하고, 이를 텍스트로 변환하는 방식'
    }
    
    # 매니페스트 저장
    output_dir = "data"
    ensure_dir(output_dir)
    
    # 전체 매니페스트
    full_manifest_path = os.path.join(output_dir, "manifest_pronunciation_word_30.json")
    save_json(full_manifest, full_manifest_path)
    
    # 개별 매니페스트
    train_manifest_path = os.path.join(output_dir, "manifest_pronunciation_word_30_train.json")
    val_manifest_path = os.path.join(output_dir, "manifest_pronunciation_word_30_val.json")
    
    save_json(train_data, train_manifest_path)
    save_json(val_data, val_manifest_path)
    
    print(f"실제 발음 기반 30개 단어 매니페스트 생성 완료:")
    print(f"  - 전체 매니페스트: {full_manifest_path}")
    print(f"  - 훈련 매니페스트: {train_manifest_path}")
    print(f"  - 검증 매니페스트: {val_manifest_path}")
    print(f"  - 훈련: {len(train_data)}개, 검증: {len(val_data)}개")
    print(f"  - 학습 방식: 음성 → 실제 발음 IPA → 텍스트")
    
    # 발음 예시 출력
    print("\n발음 기반 학습 예시:")
    print("  기대 텍스트 → 실제 발음 → IPA")
    
    for i, item in enumerate(manifest_data[:10]):
        print(f"  {item['expected_text']:8s} → {item['actual_pronunciation']:8s} → {item['actual_ipa']}")
    
    # 어려운 발음들 출력
    hard_pronunciations = [item for item in manifest_data if item['difficulty'] == 'hard']
    if hard_pronunciations:
        print(f"\n어려운 발음들 (총 {len(hard_pronunciations)}개):")
        for item in hard_pronunciations:
            print(f"  {item['expected_text']:8s} → {item['actual_pronunciation']:8s} → {item['actual_ipa']}")
    
    return full_manifest_path, train_manifest_path, val_manifest_path

def create_custom_pronunciation_manifest(custom_pronunciations: List[tuple]):
    """사용자 정의 발음으로 매니페스트를 생성합니다."""
    
    # IPA 변환기 초기화
    ipa_converter = KoreanToIPA()
    
    # 매니페스트 데이터 생성
    manifest_data = []
    
    for i, (expected_text, actual_pronunciation) in enumerate(custom_pronunciations, 1):
        # 오디오 파일 경로
        audio_file = f"data/raw/wav/custom_{i:02d}_{expected_text}.wav"
        
        # 전사본 파일 경로
        transcript_file = f"data/raw/transcripts/custom_{i:02d}_{expected_text}.txt"
        
        # 실제 발음을 IPA로 변환
        try:
            actual_ipa = ipa_converter.text_to_ipa(actual_pronunciation)
        except Exception as e:
            print(f"IPA 변환 오류 ({actual_pronunciation}): {e}")
            actual_ipa = actual_pronunciation
        
        # 기대하는 텍스트를 IPA로 변환
        try:
            expected_ipa = ipa_converter.text_to_ipa(expected_text)
        except Exception as e:
            print(f"IPA 변환 오류 ({expected_text}): {e}")
            expected_ipa = expected_text
        
        # 매니페스트 항목 생성
        manifest_item = {
            'audio_file': audio_file,
            'transcript_file': transcript_file,
            'expected_text': expected_text,
            'actual_pronunciation': actual_pronunciation,
            'actual_ipa': actual_ipa,
            'expected_ipa': expected_ipa,
            'word_id': i,
            'word': expected_text,
            'duration': None,
            'category': 'custom_pronunciation',
            'difficulty': 'custom'
        }
        
        manifest_data.append(manifest_item)
    
    # 매니페스트 저장
    output_dir = "data"
    ensure_dir(output_dir)
    
    custom_manifest_path = os.path.join(output_dir, "manifest_custom_pronunciation.json")
    save_json(manifest_data, custom_manifest_path)
    
    print(f"사용자 정의 발음 매니페스트 생성 완료:")
    print(f"  - 매니페스트: {custom_manifest_path}")
    print(f"  - 총 {len(manifest_data)}개 단어")
    
    # 발음 예시 출력
    print("\n사용자 정의 발음:")
    for item in manifest_data:
        print(f"  {item['expected_text']:8s} → {item['actual_pronunciation']:8s} → {item['actual_ipa']}")
    
    return custom_manifest_path

def main():
    parser = argparse.ArgumentParser(description="실제 발음 기반 매니페스트 생성")
    parser.add_argument(
        '--custom', 
        nargs='+',
        help='사용자 정의 발음 (예: --custom 바지:아지 가방:가방)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data',
        help='출력 디렉토리'
    )
    
    args = parser.parse_args()
    
    if args.custom:
        # 사용자 정의 발음 파싱
        custom_pronunciations = []
        for item in args.custom:
            if ':' in item:
                expected, actual = item.split(':', 1)
                custom_pronunciations.append((expected.strip(), actual.strip()))
            else:
                print(f"잘못된 형식: {item} (예상 형식: 바지:아지)")
                return 1
        
        # 사용자 정의 발음 매니페스트 생성
        create_custom_pronunciation_manifest(custom_pronunciations)
    else:
        # 기본 30개 단어 발음 매니페스트 생성
        create_pronunciation_manifest()
    
    return 0

if __name__ == "__main__":
    exit(main()) 