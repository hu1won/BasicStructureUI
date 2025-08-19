"""
한국어 텍스트를 자모로 분해하는 Grapheme-to-Phoneme 변환 모듈
"""

import re
from typing import List, Tuple

# 한글 유니코드 범위
HANGUL_START = 0xAC00
HANGUL_END = 0xD7AF

# 자음과 모음의 개수
CONSONANT_COUNT = 19
VOWEL_COUNT = 21

# 초성 자음 리스트
INITIAL_CONSONANTS = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# 중성 모음 리스트
MEDIAL_VOWELS = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]

# 종성 자음 리스트
FINAL_CONSONANTS = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

def is_hangul(char: str) -> bool:
    """문자가 한글인지 확인합니다.
    
    Args:
        char (str): 확인할 문자
    
    Returns:
        bool: 한글 여부
    """
    return HANGUL_START <= ord(char) <= HANGUL_END

def decompose_hangul(char: str) -> Tuple[str, str, str]:
    """한글 문자를 초성, 중성, 종성으로 분해합니다.
    
    Args:
        char (str): 한글 문자
    
    Returns:
        Tuple[str, str, str]: (초성, 중성, 종성)
    """
    if not is_hangul(char):
        return char, '', ''
    
    # 한글 유니코드에서 초성, 중성, 종성 계산
    code = ord(char) - HANGUL_START
    
    final_index = code % 28
    vowel_index = (code // 28) % VOWEL_COUNT
    consonant_index = code // (28 * VOWEL_COUNT)
    
    initial = INITIAL_CONSONANTS[consonant_index]
    medial = MEDIAL_VOWELS[vowel_index]
    final = FINAL_CONSONANTS[final_index] if final_index > 0 else ''
    
    return initial, medial, final

def text_to_jamo(text: str) -> List[str]:
    """한국어 텍스트를 자모 단위로 분해합니다.
    
    Args:
        text (str): 한국어 텍스트
    
    Returns:
        List[str]: 자모 리스트
    """
    jamo_list = []
    
    for char in text:
        if is_hangul(char):
            initial, medial, final = decompose_hangul(char)
            jamo_list.extend([initial, medial])
            if final:
                jamo_list.append(final)
        else:
            jamo_list.append(char)
    
    return jamo_list

def text_to_syllables(text: str) -> List[str]:
    """한국어 텍스트를 음절 단위로 분리합니다.
    
    Args:
        text (str): 한국어 텍스트
    
    Returns:
        List[str]: 음절 리스트
    """
    # 한글과 비한글을 분리
    pattern = r'[가-힣]+|[^가-힣]+'
    syllables = re.findall(pattern, text)
    
    # 빈 문자열 제거
    return [s for s in syllables if s.strip()]

def normalize_text(text: str) -> str:
    """텍스트를 정규화합니다.
    
    Args:
        text (str): 원본 텍스트
    
    Returns:
        str: 정규화된 텍스트
    """
    # 공백 정규화
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 특수문자 처리
    text = re.sub(r'[^\w\s가-힣]', '', text)
    
    return text

def get_phoneme_features(char: str) -> dict:
    """문자의 음성학적 특징을 반환합니다.
    
    Args:
        char (str): 문자
    
    Returns:
        dict: 음성학적 특징
    """
    if not is_hangul(char):
        return {'type': 'other', 'char': char}
    
    initial, medial, final = decompose_hangul(char)
    
    features = {
        'type': 'hangul',
        'initial': initial,
        'medial': medial,
        'final': final,
        'has_final': bool(final),
        'syllable_count': 1
    }
    
    return features

if __name__ == "__main__":
    # 테스트
    test_text = "안녕하세요"
    print(f"원본 텍스트: {test_text}")
    print(f"자모 분해: {text_to_jamo(test_text)}")
    print(f"음절 분리: {text_to_syllables(test_text)}")
    
    for char in test_text:
        features = get_phoneme_features(char)
        print(f"'{char}': {features}")
