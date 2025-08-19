#!/usr/bin/env python3
"""
데이터 매니페스트를 생성하는 스크립트
오디오 파일과 전사본을 매핑하여 학습/평가용 매니페스트를 생성합니다.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import random

def create_manifest(
    audio_dir: str,
    transcript_dir: str,
    output_path: str,
    split_ratio: float = 0.8,
    seed: int = 42
) -> None:
    """데이터 매니페스트를 생성합니다.
    
    Args:
        audio_dir (str): 오디오 파일 디렉토리
        transcript_dir (str): 전사본 파일 디렉토리
        output_path (str): 출력 매니페스트 파일 경로
        split_ratio (float): 훈련/검증 분할 비율
        seed (int): 랜덤 시드
    """
    # 시드 설정
    random.seed(seed)
    
    # 디렉토리 존재 확인
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"오디오 디렉토리가 존재하지 않습니다: {audio_dir}")
    
    if not os.path.exists(transcript_dir):
        raise FileNotFoundError(f"전사본 디렉토리가 존재하지 않습니다: {transcript_dir}")
    
    # 오디오 파일과 전사본 파일 매칭
    manifest_data = []
    
    # 오디오 파일 리스트
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.m4a']:
        audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
    
    # 전사본 파일 리스트
    transcript_files = list(Path(transcript_dir).glob("*.txt"))
    
    print(f"발견된 오디오 파일: {len(audio_files)}개")
    print(f"발견된 전사본 파일: {len(transcript_files)}개")
    
    # 파일 매칭
    for audio_file in audio_files:
        # 전사본 파일 찾기 (파일명 기반)
        audio_name = audio_file.stem
        transcript_file = None
        
        for transcript in transcript_files:
            if transcript.stem == audio_name:
                transcript_file = transcript
                break
        
        if transcript_file is None:
            print(f"경고: {audio_name}에 대한 전사본을 찾을 수 없습니다.")
            continue
        
        # 전사본 내용 읽기
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                print(f"경고: {transcript_file}가 비어있습니다.")
                continue
            
            # 매니페스트 항목 생성
            manifest_item = {
                'audio_file': str(audio_file),
                'transcript_file': str(transcript_file),
                'text': text,
                'audio_name': audio_name,
                'duration': None  # 나중에 계산
            }
            
            manifest_data.append(manifest_item)
            
        except Exception as e:
            print(f"경고: {transcript_file} 읽기 실패: {e}")
            continue
    
    if not manifest_data:
        raise ValueError("매칭되는 파일이 없습니다.")
    
    print(f"매칭된 파일: {len(manifest_data)}개")
    
    # 훈련/검증 분할
    random.shuffle(manifest_data)
    split_idx = int(len(manifest_data) * split_ratio)
    
    train_data = manifest_data[:split_idx]
    val_data = manifest_data[split_idx:]
    
    # 매니페스트 저장
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 전체 매니페스트
    full_manifest = {
        'train': train_data,
        'val': val_data,
        'total_samples': len(manifest_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'split_ratio': split_ratio,
        'seed': seed
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_manifest, f, ensure_ascii=False, indent=2)
    
    print(f"매니페스트가 저장되었습니다: {output_path}")
    print(f"훈련 데이터: {len(train_data)}개")
    print(f"검증 데이터: {len(val_data)}개")
    
    # 개별 매니페스트도 저장
    train_path = output_path.replace('.json', '_train.json')
    val_path = output_path.replace('.json', '_val.json')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"훈련 매니페스트: {train_path}")
    print(f"검증 매니페스트: {val_path}")

def validate_manifest(manifest_path: str) -> Dict:
    """매니페스트의 유효성을 검사합니다.
    
    Args:
        manifest_path (str): 매니페스트 파일 경로
    
    Returns:
        Dict: 검증 결과
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_count': 0,
            'missing_files': []
        }
        
        # 파일 존재 여부 확인
        for split_name, split_data in [('train', manifest.get('train', [])), 
                                     ('val', manifest.get('val', []))]:
            for item in split_data:
                validation_result['file_count'] += 1
                
                # 오디오 파일 확인
                if not os.path.exists(item['audio_file']):
                    validation_result['errors'].append(
                        f"오디오 파일이 존재하지 않습니다: {item['audio_file']}"
                    )
                    validation_result['missing_files'].append(item['audio_file'])
                
                # 전사본 파일 확인
                if not os.path.exists(item['transcript_file']):
                    validation_result['errors'].append(
                        f"전사본 파일이 존재하지 않습니다: {item['transcript_file']}"
                    )
                    validation_result['missing_files'].append(item['transcript_file'])
                
                # 텍스트 내용 확인
                if not item.get('text', '').strip():
                    validation_result['warnings'].append(
                        f"빈 전사본: {item['transcript_file']}"
                    )
        
        if validation_result['errors']:
            validation_result['is_valid'] = False
        
        return validation_result
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"매니페스트 읽기 실패: {str(e)}"],
            'warnings': [],
            'file_count': 0,
            'missing_files': []
        }

def main():
    parser = argparse.ArgumentParser(description="데이터 매니페스트 생성")
    parser.add_argument(
        '--audio_dir', 
        type=str, 
        required=True,
        help='오디오 파일 디렉토리'
    )
    parser.add_argument(
        '--transcript_dir', 
        type=str, 
        required=True,
        help='전사본 파일 디렉토리'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/manifest.json',
        help='출력 매니페스트 파일 경로'
    )
    parser.add_argument(
        '--split_ratio', 
        type=float, 
        default=0.8,
        help='훈련/검증 분할 비율 (기본값: 0.8)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='랜덤 시드 (기본값: 42)'
    )
    parser.add_argument(
        '--validate', 
        action='store_true',
        help='생성된 매니페스트 검증'
    )
    
    args = parser.parse_args()
    
    try:
        # 매니페스트 생성
        create_manifest(
            audio_dir=args.audio_dir,
            transcript_dir=args.transcript_dir,
            output_path=args.output,
            split_ratio=args.split_ratio,
            seed=args.seed
        )
        
        # 검증 (선택사항)
        if args.validate:
            print("\n=== 매니페스트 검증 ===")
            validation_result = validate_manifest(args.output)
            
            if validation_result['is_valid']:
                print("✅ 매니페스트 검증 통과")
            else:
                print("❌ 매니페스트 검증 실패")
                for error in validation_result['errors']:
                    print(f"  오류: {error}")
            
            for warning in validation_result['warnings']:
                print(f"  경고: {warning}")
            
            print(f"총 파일 수: {validation_result['file_count']}")
            if validation_result['missing_files']:
                print(f"누락된 파일: {len(validation_result['missing_files'])}개")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
