#!/usr/bin/env python3
"""
오디오 파일명을 매니페스트와 일치하도록 변경하는 스크립트
"""

import os
import shutil

# 30개 단어 목록 (매니페스트 순서대로)
words = [
    "바지", "가방", "접시", "장갑", "뽀뽀", "포크", "아프다", "단추", "침대", "숟가락",
    "꽃", "딸기", "목도리", "토끼", "코", "짹짹", "사탕", "우산", "싸우다", "눈사람",
    "휴지", "비행기", "먹다", "라면", "나무", "그네", "양말", "머리", "나비", "웃다"
]

def rename_audio_files():
    """오디오 파일명을 변경합니다."""
    wav_dir = "data/raw/wav"
    
    if not os.path.exists(wav_dir):
        print(f"❌ WAV 디렉토리가 존재하지 않습니다: {wav_dir}")
        return
    
    print("🔄 오디오 파일명 변경 시작...")
    
    # 현재 파일 목록 확인
    current_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    print(f"📁 현재 WAV 파일 수: {len(current_files)}")
    
    # 파일명 변경
    renamed_count = 0
    for i, word in enumerate(words, 1):
        old_name = f"{word}.wav"
        new_name = f"word_{i:02d}_{word}.wav"
        
        old_path = os.path.join(wav_dir, old_name)
        new_path = os.path.join(wav_dir, new_name)
        
        if os.path.exists(old_path):
            try:
                shutil.move(old_path, new_path)
                print(f"✅ {old_name} → {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"❌ {old_name} 변경 실패: {e}")
        else:
            print(f"⚠️ 파일이 존재하지 않음: {old_name}")
    
    print(f"\n🎉 파일명 변경 완료: {renamed_count}/{len(words)}개 파일")
    
    # 변경 후 파일 목록 확인
    final_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    print(f"📁 최종 WAV 파일 수: {len(final_files)}")
    
    # 변경된 파일명 출력
    print("\n📋 변경된 파일명:")
    for f in sorted(final_files):
        print(f"  - {f}")

if __name__ == "__main__":
    rename_audio_files() 