"""
한국어 자모를 IPA(국제음성기호)로 매핑하는 모듈
"""

# 자음 매핑 (초성/종성 구분)
CONSONANTS = {
    # 초성 자음
    'ㄱ': 'k', 'ㄲ': 'k*', 'ㄴ': 'n', 'ㄷ': 't', 'ㄸ': 't*',
    'ㄹ': 'l', 'ㅁ': 'm', 'ㅂ': 'p', 'ㅃ': 'p*', 'ㅅ': 's',
    'ㅆ': 's*', 'ㅇ': 'ŋ', 'ㅈ': 'tʃ', 'ㅉ': 'tʃ*', 'ㅊ': 'tʃʰ',
    'ㅋ': 'kʰ', 'ㅌ': 'tʰ', 'ㅍ': 'pʰ', 'ㅎ': 'h',
    
    # 종성 자음 (받침)
    'ㄱ_': 'k', 'ㄲ_': 'k*', 'ㄴ_': 'n', 'ㄷ_': 't', 'ㄹ_': 'l',
    'ㅁ_': 'm', 'ㅂ_': 'p', 'ㅅ_': 't', 'ㅆ_': 't', 'ㅇ_': 'ŋ',
    'ㅈ_': 't', 'ㅊ_': 't', 'ㅋ_': 'k', 'ㅌ_': 't', 'ㅍ_': 'p', 'ㅎ_': 't'
}

# 모음 매핑
VOWELS = {
    'ㅏ': 'a', 'ㅐ': 'ɛ', 'ㅑ': 'ja', 'ㅒ': 'jɛ', 'ㅓ': 'ʌ',
    'ㅔ': 'e', 'ㅕ': 'jʌ', 'ㅖ': 'je', 'ㅗ': 'o', 'ㅘ': 'wa',
    'ㅙ': 'wɛ', 'ㅚ': 'we', 'ㅛ': 'jo', 'ㅜ': 'u', 'ㅝ': 'wʌ',
    'ㅞ': 'we', 'ㅟ': 'wi', 'ㅠ': 'ju', 'ㅡ': 'ɯ', 'ㅢ': 'ɯi',
    'ㅣ': 'i'
}

# 이중 자음 매핑
DOUBLE_CONSONANTS = {
    'ㄳ': 'ks', 'ㄵ': 'ntʃ', 'ㄶ': 'nh', 'ㄺ': 'lk', 'ㄻ': 'lm',
    'ㄼ': 'lp', 'ㄽ': 'ls', 'ㄾ': 'ltʰ', 'ㄿ': 'lpʰ', 'ㅀ': 'lh',
    'ㅄ': 'ps'
}

# 음소 변환 규칙
PHONEME_RULES = {
    # 자음 동화 규칙
    'ㄱ+ㄴ': 'ŋn', 'ㄱ+ㄹ': 'ŋl', 'ㄱ+ㅁ': 'ŋm',
    'ㄷ+ㄴ': 'nn', 'ㄷ+ㄹ': 'll', 'ㄷ+ㅁ': 'mm',
    'ㅂ+ㄴ': 'mn', 'ㅂ+ㄹ': 'ml', 'ㅂ+ㅁ': 'mm',
    
    # 모음 조화 규칙
    'ㅏ+ㅣ': 'ai', 'ㅓ+ㅣ': 'ʌi', 'ㅗ+ㅣ': 'oi', 'ㅜ+ㅣ': 'ui'
}

def get_consonant_ipa(char, position='initial'):
    """자음의 IPA를 반환합니다.
    
    Args:
        char (str): 한국어 자음
        position (str): 'initial' (초성) 또는 'final' (종성)
    
    Returns:
        str: IPA 자음
    """
    if position == 'final':
        key = f"{char}_"
        return CONSONANTS.get(key, CONSONANTS.get(char, char))
    return CONSONANTS.get(char, char)

def get_vowel_ipa(char):
    """모음의 IPA를 반환합니다.
    
    Args:
        char (str): 한국어 모음
    
    Returns:
        str: IPA 모음
    """
    return VOWELS.get(char, char)

def get_double_consonant_ipa(char):
    """이중 자음의 IPA를 반환합니다.
    
    Args:
        char (str): 한국어 이중 자음
    
    Returns:
        str: IPA 자음 조합
    """
    return DOUBLE_CONSONANTS.get(char, char)
