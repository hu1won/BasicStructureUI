"""
한국어 텍스트를 IPA(국제음성기호)로 변환하는 메인 모듈
"""

from typing import List, Dict, Optional
from .g2p_ko import text_to_jamo, text_to_syllables, normalize_text
from .ipa_map import get_consonant_ipa, get_vowel_ipa, get_double_consonant_ipa
from .postrules import PostProcessor, apply_post_rules

class KoreanToIPA:
    """한국어 텍스트를 IPA로 변환하는 클래스"""
    
    def __init__(self):
        self.post_processor = PostProcessor()
        self.ipa_cache = {}  # 변환 결과 캐시
    
    def text_to_ipa(self, text: str, apply_post_rules: bool = True, 
                    stress_type: str = 'initial') -> str:
        """한국어 텍스트를 IPA로 변환합니다.
        
        Args:
            text (str): 한국어 텍스트
            apply_post_rules (bool): 후처리 규칙 적용 여부
            stress_type (str): 강세 타입 ('initial', 'final', 'none')
        
        Returns:
            str: IPA 텍스트
        """
        # 캐시 확인
        cache_key = f"{text}_{apply_post_rules}_{stress_type}"
        if cache_key in self.ipa_cache:
            return self.ipa_cache[cache_key]
        
        # 텍스트 정규화
        normalized_text = normalize_text(text)
        
        # 자모로 분해
        jamo_list = text_to_jamo(normalized_text)
        
        # IPA로 변환
        ipa_list = self._jamo_to_ipa(jamo_list)
        
        # IPA 텍스트 조합
        ipa_text = ''.join(ipa_list)
        
        # 후처리 규칙 적용
        if apply_post_rules:
            ipa_text = self.post_processor.process(ipa_text, apply_stress=True)
            
            # 강세 타입에 따른 추가 처리
            if stress_type == 'final':
                ipa_text = self.post_processor.apply_stress(ipa_text, 'final')
            elif stress_type == 'none':
                # 강세 제거
                ipa_text = ipa_text.replace('ˈ', '').replace('ˌ', '')
        
        # 결과 캐시
        self.ipa_cache[cache_key] = ipa_text
        
        return ipa_text
    
    def _jamo_to_ipa(self, jamo_list: List[str]) -> List[str]:
        """자모 리스트를 IPA로 변환합니다.
        
        Args:
            jamo_list (List[str]): 자모 리스트
        
        Returns:
            List[str]: IPA 리스트
        """
        ipa_list = []
        i = 0
        
        while i < len(jamo_list):
            current = jamo_list[i]
            
            # 이중 자음 처리
            if i + 1 < len(jamo_list):
                double_char = current + jamo_list[i + 1]
                if double_char in ['ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']:
                    ipa_list.append(get_double_consonant_ipa(double_char))
                    i += 2
                    continue
            
            # 일반 자음/모음 처리
            if current in ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']:
                # 자음인 경우
                if i == 0 or (i > 0 and jamo_list[i-1] in ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']):
                    # 초성 자음
                    ipa_list.append(get_consonant_ipa(current, 'initial'))
                else:
                    # 종성 자음
                    ipa_list.append(get_consonant_ipa(current, 'final'))
            elif current in ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']:
                # 모음인 경우
                ipa_list.append(get_vowel_ipa(current))
            else:
                # 기타 문자 (영문, 숫자, 특수문자 등)
                ipa_list.append(current)
            
            i += 1
        
        return ipa_list
    
    def batch_convert(self, texts: List[str], **kwargs) -> List[str]:
        """여러 텍스트를 일괄 변환합니다.
        
        Args:
            texts (List[str]): 한국어 텍스트 리스트
            **kwargs: text_to_ipa 메서드의 인자들
        
        Returns:
            List[str]: IPA 텍스트 리스트
        """
        return [self.text_to_ipa(text, **kwargs) for text in texts]
    
    def get_conversion_stats(self, text: str) -> Dict:
        """변환 통계를 반환합니다.
        
        Args:
            text (str): 한국어 텍스트
        
        Returns:
            Dict: 변환 통계
        """
        jamo_list = text_to_jamo(text)
        syllables = text_to_syllables(text)
        
        stats = {
            'original_length': len(text),
            'jamo_count': len(jamo_list),
            'syllable_count': len(syllables),
            'consonant_count': sum(1 for j in jamo_list if j in ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']),
            'vowel_count': sum(1 for j in jamo_list if j in ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'])
        }
        
        return stats
    
    def clear_cache(self):
        """변환 결과 캐시를 클리어합니다."""
        self.ipa_cache.clear()

def text_to_ipa(text: str, **kwargs) -> str:
    """한국어 텍스트를 IPA로 변환하는 편의 함수입니다.
    
    Args:
        text (str): 한국어 텍스트
        **kwargs: KoreanToIPA.text_to_ipa 메서드의 인자들
    
    Returns:
        str: IPA 텍스트
    """
    converter = KoreanToIPA()
    return converter.text_to_ipa(text, **kwargs)

if __name__ == "__main__":
    # 테스트
    converter = KoreanToIPA()
    
    test_texts = [
        "안녕하세요",
        "커피 한 잔 마실까요?",
        "오늘 날씨가 좋네요",
        "감사합니다"
    ]
    
    print("=== 한국어 → IPA 변환 테스트 ===")
    for text in test_texts:
        ipa_result = converter.text_to_ipa(text)
        stats = converter.get_conversion_stats(text)
        print(f"원본: {text}")
        print(f"IPA: {ipa_result}")
        print(f"통계: {stats}")
        print("-" * 30)
    
    # 배치 변환 테스트
    print("\n=== 배치 변환 테스트 ===")
    batch_results = converter.batch_convert(test_texts)
    for original, ipa in zip(test_texts, batch_results):
        print(f"{original} → {ipa}")
