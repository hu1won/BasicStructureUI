#!/usr/bin/env python3
"""
매니페스트 파일의 경로 수정 스크립트
"""

import json

def fix_manifest_paths(manifest_file):
    """매니페스트 파일의 경로를 수정합니다"""
    print(f"📋 매니페스트 파일 수정 중: {manifest_file}")
    
    # 매니페스트 로드
    with open(manifest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 경로 수정
    for item in data:
        # audio_file 경로 수정
        if 'audio_file' in item:
            old_path = item['audio_file']
            # data/raw/wav/ → data/raw/wav/
            new_path = old_path.replace('data/raw/wav/', 'data/raw/wav/')
            item['audio_file'] = new_path
            print(f"  📁 {old_path} → {new_path}")
    
    # 매니페스트 저장
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 매니페스트 수정 완료: {manifest_file}")

def main():
    """메인 함수"""
    print("=== 매니페스트 경로 수정 시작 ===\n")
    
    # 훈련 매니페스트 수정
    fix_manifest_paths("data/manifest_pronunciation_word_30_train.json")
    
    # 검증 매니페스트 수정
    fix_manifest_paths("data/manifest_pronunciation_word_30_val.json")
    
    print("\n🎉 모든 매니페스트 경로 수정 완료!")

if __name__ == "__main__":
    main() 