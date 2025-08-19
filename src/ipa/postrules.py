"""
IPA 변환 후 적용할 후처리 규칙들
"""

import re
from typing import List, Dict, Tuple

class PostProcessor:
    """IPA 후처리 규칙을 적용하는 클래스"""
    
    def __init__(self):
        # 자음 동화 규칙
        self.consonant_assimilation = {
            # 비음화 (nasalization)
            'k+n': 'ŋn', 'k+l': 'ŋl', 'k+m': 'ŋm',
            't+n': 'nn', 't+l': 'll', 't+m': 'mm',
            'p+n': 'mn', 'p+l': 'ml', 'p+m': 'mm',
            
            # 유음화 (liquidization)
            'n+l': 'll', 'l+n': 'nn',
            
            # 파열음화 (plosivization)
            'ŋ+k': 'ŋk', 'm+p': 'mp', 'n+t': 'nt'
        }
        
        # 모음 조화 규칙
        self.vowel_harmony = {
            'a+i': 'ai', 'ʌ+i': 'ʌi', 'o+i': 'oi', 'u+i': 'ui',
            'a+u': 'au', 'e+u': 'eu', 'o+u': 'ou'
        }
        
        # 음절 경계 규칙
        self.syllable_boundary = {
            'k.k': 'k.k', 't.t': 't.t', 'p.p': 'p.p',
            'k.t': 'k.t', 't.p': 't.p', 'p.k': 'p.k'
        }
        
        # 강세 규칙
        self.stress_rules = {
            'initial': 'ˈ',  # 첫 음절 강세
            'final': 'ˌ'     # 마지막 음절 강세
        }
    
    def apply_consonant_assimilation(self, ipa_text: str) -> str:
        """자음 동화 규칙을 적용합니다.
        
        Args:
            ipa_text (str): IPA 텍스트
        
        Returns:
            str: 자음 동화가 적용된 IPA 텍스트
        """
        result = ipa_text
        
        for pattern, replacement in self.consonant_assimilation.items():
            # 정규식으로 패턴 매칭
            regex_pattern = pattern.replace('+', r'\+')
            result = re.sub(regex_pattern, replacement, result)
        
        return result
    
    def apply_vowel_harmony(self, ipa_text: str) -> str:
        """모음 조화 규칙을 적용합니다.
        
        Args:
            ipa_text (str): IPA 텍스트
        
        Returns:
            str: 모음 조화가 적용된 IPA 텍스트
        """
        result = ipa_text
        
        for pattern, replacement in self.vowel_harmony.items():
            regex_pattern = pattern.replace('+', r'\+')
            result = re.sub(regex_pattern, replacement, result)
        
        return result
    
    def apply_syllable_boundary(self, ipa_text: str) -> str:
        """음절 경계 규칙을 적용합니다.
        
        Args:
            ipa_text (str): IPA 텍스트
        
        Returns:
            str: 음절 경계가 적용된 IPA 텍스트
        """
        result = ipa_text
        
        # 음절 경계를 '.'으로 표시
        for pattern, replacement in self.syllable_boundary.items():
            regex_pattern = pattern.replace('.', r'\.')
            result = re.sub(regex_pattern, replacement, result)
        
        return result
    
    def apply_stress(self, ipa_text: str, stress_type: str = 'initial') -> str:
        """강세 규칙을 적용합니다.
        
        Args:
            ipa_text (str): IPA 텍스트
            stress_type (str): 강세 타입 ('initial' 또는 'final')
        
        Returns:
            str: 강세가 적용된 IPA 텍스트
        """
        if stress_type not in self.stress_rules:
            return ipa_text
        
        stress_mark = self.stress_rules[stress_type]
        
        if stress_type == 'initial':
            # 첫 음절에 강세 표시
            syllables = ipa_text.split('.')
            if syllables:
                syllables[0] = stress_mark + syllables[0]
            return '.'.join(syllables)
        
        elif stress_type == 'final':
            # 마지막 음절에 강세 표시
            syllables = ipa_text.split('.')
            if syllables:
                syllables[-1] = stress_mark + syllables[-1]
            return '.'.join(syllables)
        
        return ipa_text
    
    def normalize_ipa(self, ipa_text: str) -> str:
        """IPA 텍스트를 정규화합니다.
        
        Args:
            ipa_text (str): IPA 텍스트
        
        Returns:
            str: 정규화된 IPA 텍스트
        """
        # 연속된 공백 제거
        result = re.sub(r'\s+', ' ', ipa_text.strip())
        
        # 불필요한 기호 제거
        result = re.sub(r'[^\w\s\.ˈˌ]', '', result)
        
        # 음절 경계 정규화
        result = re.sub(r'\.+', '.', result)
        
        return result
    
    def process(self, ipa_text: str, apply_stress: bool = True) -> str:
        """모든 후처리 규칙을 적용합니다.
        
        Args:
            ipa_text (str): IPA 텍스트
            apply_stress (bool): 강세 적용 여부
        
        Returns:
            str: 후처리가 완료된 IPA 텍스트
        """
        result = ipa_text
        
        # 자음 동화 적용
        result = self.apply_consonant_assimilation(result)
        
        # 모음 조화 적용
        result = self.apply_vowel_harmony(result)
        
        # 음절 경계 적용
        result = self.apply_syllable_boundary(result)
        
        # 강세 적용
        if apply_stress:
            result = self.apply_stress(result, 'initial')
        
        # 정규화
        result = self.normalize_ipa(result)
        
        return result

def apply_post_rules(ipa_text: str, **kwargs) -> str:
    """후처리 규칙을 적용하는 편의 함수입니다.
    
    Args:
        ipa_text (str): IPA 텍스트
        **kwargs: PostProcessor.process() 메서드의 인자들
    
    Returns:
        str: 후처리가 완료된 IPA 텍스트
    """
    processor = PostProcessor()
    return processor.process(ipa_text, **kwargs)

if __name__ == "__main__":
    # 테스트
    processor = PostProcessor()
    
    test_ipa = "kanna"
    print(f"원본 IPA: {test_ipa}")
    
    # 자음 동화 적용
    result = processor.apply_consonant_assimilation(test_ipa)
    print(f"자음 동화 후: {result}")
    
    # 전체 후처리 적용
    final_result = processor.process(test_ipa)
    print(f"최종 결과: {final_result}")
