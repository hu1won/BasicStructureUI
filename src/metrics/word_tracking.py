"""
단어별 성능 추적을 위한 메트릭 모듈
30개 단어의 개별 인식 성능을 추적하고 분석합니다.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

from .asr_metrics import calculate_wer, calculate_cer

class WordPerformanceTracker:
    """단어별 성능을 추적하는 클래스"""
    
    def __init__(self, target_words: List[str]):
        """단어 성능 추적기를 초기화합니다.
        
        Args:
            target_words (List[str]): 추적할 대상 단어 리스트
        """
        self.target_words = target_words
        self.word_to_idx = {word: i for i, word in enumerate(target_words)}
        
        # 단어별 성능 통계 초기화
        self.word_stats = defaultdict(lambda: {
            'total_attempts': 0,
            'correct_recognitions': 0,
            'total_wer': 0.0,
            'total_cer': 0.0,
            'confusion_matrix': defaultdict(int),
            'recognition_history': []
        })
        
        # 전체 성능 통계
        self.overall_stats = {
            'total_samples': 0,
            'total_correct': 0,
            'overall_wer': 0.0,
            'overall_cer': 0.0
        }
    
    def update(self, reference: str, prediction: str, sample_id: str = None):
        """단어 인식 결과를 업데이트합니다.
        
        Args:
            reference (str): 참조 텍스트
            prediction (str): 예측 텍스트
            sample_id (str): 샘플 식별자
        """
        # 참조 단어가 대상 단어인지 확인
        ref_words = reference.strip().split()
        pred_words = prediction.strip().split()
        
        for ref_word in ref_words:
            if ref_word in self.target_words:
                # 단어별 통계 업데이트
                self._update_word_stats(ref_word, ref_word, pred_words, sample_id)
        
        # 전체 성능 업데이트
        self._update_overall_stats(reference, prediction)
    
    def _update_word_stats(self, target_word: str, reference: str, 
                          prediction_words: List[str], sample_id: str = None):
        """특정 단어의 통계를 업데이트합니다."""
        stats = self.word_stats[target_word]
        
        # 시도 횟수 증가
        stats['total_attempts'] += 1
        
        # 정확한 인식 여부 확인
        is_correct = target_word in prediction_words
        if is_correct:
            stats['correct_recognitions'] += 1
        
        # WER, CER 계산
        wer = calculate_wer(reference, ' '.join(prediction_words))
        cer = calculate_cer(reference, ' '.join(prediction_words))
        
        stats['total_wer'] += wer
        stats['total_cer'] += cer
        
        # 혼동 행렬 업데이트
        if not is_correct and prediction_words:
            # 가장 유사한 예측 단어 찾기
            best_match = self._find_best_match(target_word, prediction_words)
            stats['confusion_matrix'][best_match] += 1
        
        # 인식 히스토리 기록
        history_entry = {
            'sample_id': sample_id,
            'reference': reference,
            'prediction': ' '.join(prediction_words),
            'is_correct': is_correct,
            'wer': wer,
            'cer': cer
        }
        stats['recognition_history'].append(history_entry)
    
    def _update_overall_stats(self, reference: str, prediction: str):
        """전체 성능 통계를 업데이트합니다."""
        self.overall_stats['total_samples'] += 1
        
        # WER, CER 계산
        wer = calculate_wer(reference, prediction)
        cer = calculate_cer(reference, prediction)
        
        self.overall_stats['overall_wer'] += wer
        self.overall_stats['overall_cer'] += cer
        
        # 정확한 인식 여부
        if wer == 0.0:
            self.overall_stats['total_correct'] += 1
    
    def _find_best_match(self, target_word: str, prediction_words: List[str]) -> str:
        """예측 단어 중에서 가장 유사한 단어를 찾습니다."""
        if not prediction_words:
            return '<empty>'
        
        # 간단한 유사도 계산 (편집 거리 기반)
        best_match = prediction_words[0]
        min_distance = float('inf')
        
        for pred_word in prediction_words:
            distance = self._levenshtein_distance(target_word, pred_word)
            if distance < min_distance:
                min_distance = distance
                best_match = pred_word
        
        return best_match
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """두 문자열 간의 편집 거리를 계산합니다."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_word_performance(self, word: str) -> Dict:
        """특정 단어의 성능을 반환합니다."""
        if word not in self.word_stats:
            return None
        
        stats = self.word_stats[word]
        
        if stats['total_attempts'] == 0:
            return {
                'word': word,
                'accuracy': 0.0,
                'wer': 0.0,
                'cer': 0.0,
                'total_attempts': 0,
                'correct_recognitions': 0
            }
        
        return {
            'word': word,
            'accuracy': stats['correct_recognitions'] / stats['total_attempts'],
            'wer': stats['total_wer'] / stats['total_attempts'],
            'cer': stats['total_cer'] / stats['total_attempts'],
            'total_attempts': stats['total_attempts'],
            'correct_recognitions': stats['correct_recognitions'],
            'confusion_matrix': dict(stats['confusion_matrix'])
        }
    
    def get_overall_performance(self) -> Dict:
        """전체 성능을 반환합니다."""
        if self.overall_stats['total_samples'] == 0:
            return {
                'overall_accuracy': 0.0,
                'overall_wer': 0.0,
                'overall_cer': 0.0,
                'total_samples': 0
            }
        
        return {
            'overall_accuracy': self.overall_stats['total_correct'] / self.overall_stats['total_samples'],
            'overall_wer': self.overall_stats['overall_wer'] / self.overall_stats['total_samples'],
            'overall_cer': self.overall_stats['overall_cer'] / self.overall_stats['total_samples'],
            'total_samples': self.overall_stats['total_samples']
        }
    
    def get_word_ranking(self) -> List[Tuple[str, float]]:
        """단어별 정확도를 기준으로 순위를 반환합니다."""
        performances = []
        
        for word in self.target_words:
            perf = self.get_word_performance(word)
            if perf:
                performances.append((word, perf['accuracy']))
        
        # 정확도 기준으로 내림차순 정렬
        performances.sort(key=lambda x: x[1], reverse=True)
        return performances
    
    def get_difficult_words(self, threshold: float = 0.5) -> List[str]:
        """어려운 단어들을 반환합니다 (정확도가 임계값 이하)."""
        difficult_words = []
        
        for word in self.target_words:
            perf = self.get_word_performance(word)
            if perf and perf['accuracy'] < threshold:
                difficult_words.append(word)
        
        return difficult_words
    
    def generate_performance_report(self) -> str:
        """성능 보고서를 생성합니다."""
        report = []
        report.append("=== 단어별 성능 보고서 ===\n")
        
        # 전체 성능
        overall = self.get_overall_performance()
        report.append(f"전체 성능:")
        report.append(f"  - 정확도: {overall['overall_accuracy']:.4f}")
        report.append(f"  - WER: {overall['overall_wer']:.4f}")
        report.append(f"  - CER: {overall['overall_cer']:.4f}")
        report.append(f"  - 총 샘플: {overall['total_samples']}\n")
        
        # 단어별 성능
        report.append("단어별 성능:")
        word_ranking = self.get_word_ranking()
        
        for i, (word, accuracy) in enumerate(word_ranking, 1):
            perf = self.get_word_performance(word)
            report.append(f"  {i:2d}. {word:8s}: {accuracy:.4f} "
                        f"({perf['correct_recognitions']:3d}/{perf['total_attempts']:3d})")
        
        # 어려운 단어
        difficult_words = self.get_difficult_words()
        if difficult_words:
            report.append(f"\n어려운 단어들 (정확도 < 0.5):")
            for word in difficult_words:
                report.append(f"  - {word}")
        
        return "\n".join(report)
    
    def save_performance_data(self, output_path: str):
        """성능 데이터를 파일로 저장합니다."""
        data = {
            'target_words': self.target_words,
            'word_stats': dict(self.word_stats),
            'overall_stats': self.overall_stats,
            'word_ranking': self.get_word_ranking(),
            'difficult_words': self.get_difficult_words()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"성능 데이터가 저장되었습니다: {output_path}")
    
    def plot_word_performance(self, output_path: str = None):
        """단어별 성능을 시각화합니다."""
        words = []
        accuracies = []
        wers = []
        cers = []
        
        for word in self.target_words:
            perf = self.get_word_performance(word)
            if perf:
                words.append(word)
                accuracies.append(perf['accuracy'])
                wers.append(perf['wer'])
                cers.append(perf['cer'])
        
        # 그래프 생성
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # 정확도
        bars1 = ax1.bar(range(len(words)), accuracies, color='skyblue')
        ax1.set_title('단어별 인식 정확도')
        ax1.set_ylabel('정확도')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(words)))
        ax1.set_xticklabels(words, rotation=45, ha='right')
        
        # WER
        bars2 = ax2.bar(range(len(words)), wers, color='lightcoral')
        ax2.set_title('단어별 Word Error Rate')
        ax2.set_ylabel('WER')
        ax2.set_ylim(0, max(wers) * 1.1 if wers else 1)
        ax2.set_xticks(range(len(words)))
        ax2.set_xticklabels(words, rotation=45, ha='right')
        
        # CER
        bars3 = ax3.bar(range(len(words)), cers, color='lightgreen')
        ax3.set_title('단어별 Character Error Rate')
        ax3.set_ylabel('CER')
        ax3.set_ylim(0, max(cers) * 1.1 if cers else 1)
        ax3.set_xticks(range(len(words)))
        ax3.set_xticklabels(words, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"성능 그래프가 저장되었습니다: {output_path}")
        
        plt.show()

def create_word_tracker(target_words: List[str]) -> WordPerformanceTracker:
    """단어 성능 추적기를 생성하는 편의 함수입니다.
    
    Args:
        target_words (List[str]): 추적할 대상 단어 리스트
    
    Returns:
        WordPerformanceTracker: 단어 성능 추적기 인스턴스
    """
    return WordPerformanceTracker(target_words)

if __name__ == "__main__":
    # 테스트
    target_words = [
        "바지", "가방", "접시", "장갑", "뽀뽀", "포크", "아프다", "단추", "침대", "숟가락",
        "꽃", "딸기", "목도리", "토끼", "코", "짹짹", "사탕", "우산", "싸우다", "눈사람",
        "휴지", "비행기", "먹다", "라면", "나무", "그네", "양말", "머리", "나비", "웃다"
    ]
    
    tracker = create_word_tracker(target_words)
    
    # 예시 데이터로 테스트
    test_cases = [
        ("바지", "바지", "correct"),
        ("가방", "가방", "correct"),
        ("접시", "젓시", "incorrect"),
        ("장갑", "장갑", "correct"),
        ("뽀뽀", "뽀뽀", "correct")
    ]
    
    for ref, pred, case_id in test_cases:
        tracker.update(ref, pred, case_id)
    
    # 성능 보고서 출력
    print(tracker.generate_performance_report())
    
    # 성능 데이터 저장
    tracker.save_performance_data("word_performance_test.json")
    
    # 그래프 생성
    tracker.plot_word_performance("word_performance_test.png") 