"""
음소 수준의 평가 메트릭
IPA 변환과 음소 인식의 정확도를 평가합니다.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import re

class PhonemeMetrics:
    """음소 수준의 평가 메트릭을 계산하는 클래스"""
    
    def __init__(self):
        """음소 메트릭 계산기를 초기화합니다."""
        self.metrics = {}
    
    def calculate_phoneme_accuracy(self, reference: str, hypothesis: str) -> Dict:
        """음소 수준의 정확도를 계산합니다.
        
        Args:
            reference (str): 참조 IPA 텍스트
            hypothesis (str): 가설 IPA 텍스트
        
        Returns:
            Dict: 음소 정확도 메트릭
        """
        # IPA 텍스트 정규화
        ref_clean = self._normalize_ipa(reference)
        hyp_clean = self._normalize_ipa(hypothesis)
        
        # 음소 단위로 분리
        ref_phonemes = self._split_phonemes(ref_clean)
        hyp_phonemes = self._split_phonemes(hyp_clean)
        
        # 정확도 계산
        total_phonemes = len(ref_phonemes)
        if total_phonemes == 0:
            return {
                'phoneme_accuracy': 0.0,
                'total_phonemes': 0,
                'correct_phonemes': 0,
                'substitution_errors': 0,
                'insertion_errors': 0,
                'deletion_errors': 0
            }
        
        # Levenshtein 거리 계산
        distance_matrix = self._levenshtein_distance(ref_phonemes, hyp_phonemes)
        
        # 오류 분석
        errors = self._analyze_phoneme_errors(ref_phonemes, hyp_phonemes)
        
        # 정확도 계산
        correct_phonemes = total_phonemes - errors['substitution'] - errors['deletion']
        phoneme_accuracy = correct_phonemes / total_phonemes
        
        return {
            'phoneme_accuracy': phoneme_accuracy,
            'total_phonemes': total_phonemes,
            'correct_phonemes': correct_phonemes,
            'substitution_errors': errors['substitution'],
            'insertion_errors': errors['insertion'],
            'deletion_errors': errors['deletion'],
            'levenshtein_distance': distance_matrix[-1][-1],
            'reference_phonemes': ref_phonemes,
            'hypothesis_phonemes': hyp_phonemes
        }
    
    def calculate_phoneme_confusion_matrix(self, references: List[str], 
                                         hypotheses: List[str]) -> Dict:
        """음소 혼동 행렬을 계산합니다.
        
        Args:
            references (List[str]): 참조 IPA 텍스트 리스트
            hypotheses (List[str]): 가설 IPA 텍스트 리스트
        
        Returns:
            Dict: 음소 혼동 행렬과 관련 메트릭
        """
        if len(references) != len(hypotheses):
            raise ValueError("참조와 가설의 개수가 일치하지 않습니다.")
        
        # 모든 음소 수집
        all_phonemes = set()
        for ref, hyp in zip(references, hypotheses):
            ref_phonemes = self._split_phonemes(self._normalize_ipa(ref))
            hyp_phonemes = self._split_phonemes(self._normalize_ipa(hyp))
            all_phonemes.update(ref_phonemes)
            all_phonemes.update(hyp_phonemes)
        
        all_phonemes = sorted(list(all_phonemes))
        phoneme_to_idx = {p: i for i, p in enumerate(all_phonemes)}
        
        # 혼동 행렬 초기화
        confusion_matrix = np.zeros((len(all_phonemes), len(all_phonemes)), dtype=int)
        
        # 각 샘플에 대해 혼동 행렬 계산
        for ref, hyp in zip(references, hypotheses):
            ref_phonemes = self._split_phonemes(self._normalize_ipa(ref))
            hyp_phonemes = self._split_phonemes(self._normalize_ipa(hyp))
            
            # 정렬된 정렬 계산
            alignment = self._align_phonemes(ref_phonemes, hyp_phonemes)
            
            for ref_phoneme, hyp_phoneme in alignment:
                if ref_phoneme and hyp_phoneme:
                    ref_idx = phoneme_to_idx[ref_phoneme]
                    hyp_idx = phoneme_to_idx[hyp_phoneme]
                    confusion_matrix[ref_idx][hyp_idx] += 1
        
        # 메트릭 계산
        metrics = self._calculate_confusion_metrics(confusion_matrix, all_phonemes)
        
        return {
            'confusion_matrix': confusion_matrix.tolist(),
            'phonemes': all_phonemes,
            'metrics': metrics,
            'total_samples': len(references)
        }
    
    def calculate_ipa_similarity(self, reference: str, hypothesis: str) -> Dict:
        """IPA 텍스트 간의 유사도를 계산합니다.
        
        Args:
            reference (str): 참조 IPA 텍스트
            hypothesis (str): 가설 IPA 텍스트
        
        Returns:
            Dict: 유사도 메트릭
        """
        # 기본 문자열 유사도
        sequence_similarity = SequenceMatcher(None, reference, hypothesis).ratio()
        
        # 음소 수준 유사도
        phoneme_metrics = self.calculate_phoneme_accuracy(reference, hypothesis)
        
        # 음소 길이 유사도
        ref_length = len(reference)
        hyp_length = len(hypothesis)
        length_similarity = 1.0 - abs(ref_length - hyp_length) / max(ref_length, hyp_length, 1)
        
        # 가중 평균 유사도
        weighted_similarity = (
            0.4 * sequence_similarity +
            0.4 * phoneme_metrics['phoneme_accuracy'] +
            0.2 * length_similarity
        )
        
        return {
            'sequence_similarity': sequence_similarity,
            'phoneme_accuracy': phoneme_metrics['phoneme_accuracy'],
            'length_similarity': length_similarity,
            'weighted_similarity': weighted_similarity,
            'phoneme_metrics': phoneme_metrics
        }
    
    def _normalize_ipa(self, ipa_text: str) -> str:
        """IPA 텍스트를 정규화합니다.
        
        Args:
            ipa_text (str): IPA 텍스트
        
        Returns:
            str: 정규화된 IPA 텍스트
        """
        # 공백 제거
        normalized = re.sub(r'\s+', '', ipa_text)
        
        # 특수 기호 정규화
        normalized = normalized.replace('ˈ', '').replace('ˌ', '')  # 강세 제거
        normalized = normalized.replace('.', '')  # 음절 경계 제거
        
        return normalized
    
    def _split_phonemes(self, ipa_text: str) -> List[str]:
        """IPA 텍스트를 음소 단위로 분리합니다.
        
        Args:
            ipa_text (str): IPA 텍스트
        
        Returns:
            List[str]: 음소 리스트
        """
        # 복합 음소 패턴 정의
        complex_phonemes = [
            'tʃ', 'dʒ', 'ŋ', 'ɲ', 'ɲ', 'ɲ', 'ɲ', 'ɲ', 'ɲ', 'ɲ',
            'wa', 'wɛ', 'we', 'jo', 'wʌ', 'wi', 'ju', 'ɯi'
        ]
        
        phonemes = []
        i = 0
        
        while i < len(ipa_text):
            found_complex = False
            
            # 복합 음소 확인
            for complex_phoneme in complex_phonemes:
                if ipa_text[i:i+len(complex_phoneme)] == complex_phoneme:
                    phonemes.append(complex_phoneme)
                    i += len(complex_phoneme)
                    found_complex = True
                    break
            
            if not found_complex:
                # 단일 음소
                phonemes.append(ipa_text[i])
                i += 1
        
        return phonemes
    
    def _levenshtein_distance(self, ref_phonemes: List[str], 
                             hyp_phonemes: List[str]) -> np.ndarray:
        """두 음소 시퀀스 간의 Levenshtein 거리를 계산합니다.
        
        Args:
            ref_phonemes (List[str]): 참조 음소 리스트
            hyp_phonemes (List[str]): 가설 음소 리스트
        
        Returns:
            np.ndarray: 거리 행렬
        """
        m, n = len(ref_phonemes), len(hyp_phonemes)
        distance_matrix = np.zeros((m + 1, n + 1), dtype=int)
        
        # 초기화
        for i in range(m + 1):
            distance_matrix[i][0] = i
        for j in range(n + 1):
            distance_matrix[0][j] = j
        
        # 거리 계산
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_phonemes[i-1] == hyp_phonemes[j-1]:
                    distance_matrix[i][j] = distance_matrix[i-1][j-1]
                else:
                    distance_matrix[i][j] = min(
                        distance_matrix[i-1][j] + 1,      # 삭제
                        distance_matrix[i][j-1] + 1,      # 삽입
                        distance_matrix[i-1][j-1] + 1     # 치환
                    )
        
        return distance_matrix
    
    def _analyze_phoneme_errors(self, ref_phonemes: List[str], 
                                hyp_phonemes: List[str]) -> Dict:
        """음소 오류를 분석합니다.
        
        Args:
            ref_phonemes (List[str]): 참조 음소 리스트
            hyp_phonemes (List[str]): 가설 음소 리스트
        
        Returns:
            Dict: 오류 분석 결과
        """
        distance_matrix = self._levenshtein_distance(ref_phonemes, hyp_phonemes)
        
        # 오류 분석을 위한 역추적
        i, j = len(ref_phonemes), len(hyp_phonemes)
        substitution = 0
        insertion = 0
        deletion = 0
        
        while i > 0 and j > 0:
            if ref_phonemes[i-1] == hyp_phonemes[j-1]:
                i -= 1
                j -= 1
            elif distance_matrix[i][j] == distance_matrix[i-1][j] + 1:
                deletion += 1
                i -= 1
            elif distance_matrix[i][j] == distance_matrix[i][j-1] + 1:
                insertion += 1
                j -= 1
            else:
                substitution += 1
                i -= 1
                j -= 1
        
        # 남은 삭제/삽입 처리
        deletion += i
        insertion += j
        
        return {
            'substitution': substitution,
            'insertion': insertion,
            'deletion': deletion
        }
    
    def _align_phonemes(self, ref_phonemes: List[str], 
                        hyp_phonemes: List[str]) -> List[Tuple[str, str]]:
        """두 음소 시퀀스를 정렬합니다.
        
        Args:
            ref_phonemes (List[str]): 참조 음소 리스트
            hyp_phonemes (List[str]): 가설 음소 리스트
        
        Returns:
            List[Tuple[str, str]]: 정렬된 음소 쌍
        """
        distance_matrix = self._levenshtein_distance(ref_phonemes, hyp_phonemes)
        
        # 정렬을 위한 역추적
        i, j = len(ref_phonemes), len(hyp_phonemes)
        alignment = []
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_phonemes[i-1] == hyp_phonemes[j-1]:
                alignment.append((ref_phonemes[i-1], hyp_phonemes[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or distance_matrix[i][j] == distance_matrix[i-1][j] + 1):
                alignment.append((ref_phonemes[i-1], None))  # 삭제
                i -= 1
            elif j > 0 and (i == 0 or distance_matrix[i][j] == distance_matrix[i][j-1] + 1):
                alignment.append((None, hyp_phonemes[j-1]))  # 삽입
                j -= 1
            else:
                alignment.append((ref_phonemes[i-1], hyp_phonemes[j-1]))  # 치환
                i -= 1
                j -= 1
        
        alignment.reverse()
        return alignment
    
    def _calculate_confusion_metrics(self, confusion_matrix: np.ndarray, 
                                   phonemes: List[str]) -> Dict:
        """혼동 행렬에서 메트릭을 계산합니다.
        
        Args:
            confusion_matrix (np.ndarray): 혼동 행렬
            phonemes (List[str]): 음소 리스트
        
        Returns:
            Dict: 계산된 메트릭
        """
        metrics = {}
        
        for i, phoneme in enumerate(phonemes):
            # True Positive, False Positive, False Negative 계산
            tp = confusion_matrix[i][i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            # 정밀도, 재현율, F1 점수 계산
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[phoneme] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positive': int(tp),
                'false_positive': int(fp),
                'false_negative': int(fn)
            }
        
        return metrics

def calculate_phoneme_metrics(reference: str, hypothesis: str) -> Dict:
    """음소 메트릭을 계산하는 편의 함수입니다.
    
    Args:
        reference (str): 참조 IPA 텍스트
        hypothesis (str): 가설 IPA 텍스트
    
    Returns:
        Dict: 음소 메트릭
    """
    metrics_calculator = PhonemeMetrics()
    return metrics_calculator.calculate_phoneme_accuracy(reference, hypothesis)

if __name__ == "__main__":
    # 테스트
    metrics_calculator = PhonemeMetrics()
    
    # 테스트 데이터
    reference = "anɲʌŋhasɛjo"
    hypothesis = "anɲʌŋhasɛjo"
    
    # 음소 정확도 계산
    accuracy_metrics = metrics_calculator.calculate_phoneme_accuracy(reference, hypothesis)
    print(f"음소 정확도: {accuracy_metrics['phoneme_accuracy']:.4f}")
    
    # IPA 유사도 계산
    similarity_metrics = metrics_calculator.calculate_ipa_similarity(reference, hypothesis)
    print(f"가중 유사도: {similarity_metrics['weighted_similarity']:.4f}")
    
    # 혼동 행렬 계산 (여러 샘플)
    references = ["anɲʌŋ", "hasɛjo", "kʌpʰi"]
    hypotheses = ["anɲʌŋ", "hasɛjo", "kʌpʰi"]
    
    confusion_result = metrics_calculator.calculate_phoneme_confusion_matrix(
        references, hypotheses
    )
    print(f"혼동 행렬 크기: {len(confusion_result['phonemes'])}x{len(confusion_result['phonemes'])}")
