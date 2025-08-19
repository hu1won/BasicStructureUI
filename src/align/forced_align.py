"""
강제 정렬(Forced Alignment)을 수행하는 메인 모듈
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .errors import (
    ForcedAlignmentError, AudioFileError, TranscriptionError,
    AlignmentError, ModelError, ValidationError, ConfigurationError,
    validate_audio_file, validate_transcription, validate_alignment_config
)

class ForcedAligner:
    """강제 정렬을 수행하는 클래스"""
    
    def __init__(self, config: Dict):
        """강제 정렬기를 초기화합니다.
        
        Args:
            config (Dict): 정렬 설정
        """
        self.config = config
        self.validate_config()
        
        # 설정에서 값 추출
        self.model_path = config.get('model_path')
        self.sample_rate = config.get('sample_rate', 16000)
        self.window_size = config.get('window_size', 0.025)
        self.hop_size = config.get('hop_size', 0.010)
        self.min_silence_duration = config.get('min_silence_duration', 0.1)
        
        # 결과 저장
        self.alignment_results = {}
    
    def validate_config(self):
        """설정의 유효성을 검사합니다."""
        try:
            validate_alignment_config(self.config)
        except ConfigurationError as e:
            raise ConfigurationError(f"설정 검증 실패: {str(e)}")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """오디오 파일을 로드합니다.
        
        Args:
            audio_path (str): 오디오 파일 경로
        
        Returns:
            np.ndarray: 오디오 데이터
        
        Raises:
            AudioFileError: 오디오 파일 로드 실패 시
        """
        try:
            validate_audio_file(audio_path)
            
            # librosa를 사용하여 오디오 로드
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(audio) == 0:
                raise AudioFileError("오디오 파일이 비어있습니다")
            
            return audio
            
        except ImportError:
            raise AudioFileError("librosa 라이브러리가 설치되지 않았습니다")
        except Exception as e:
            raise AudioFileError(f"오디오 파일 로드 실패: {str(e)}")
    
    def load_transcription(self, transcription_path: str) -> str:
        """전사본 파일을 로드합니다.
        
        Args:
            transcription_path (str): 전사본 파일 경로
        
        Returns:
            str: 전사본 텍스트
        
        Raises:
            TranscriptionError: 전사본 로드 실패 시
        """
        try:
            if not os.path.exists(transcription_path):
                raise TranscriptionError(f"전사본 파일이 존재하지 않습니다: {transcription_path}")
            
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
            
            validate_transcription(transcription)
            return transcription
            
        except Exception as e:
            raise TranscriptionError(f"전사본 로드 실패: {str(e)}")
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """오디오에서 특징을 추출합니다.
        
        Args:
            audio (np.ndarray): 오디오 데이터
        
        Returns:
            np.ndarray: 특징 벡터
        """
        try:
            import librosa
            
            # MFCC 특징 추출
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate,
                n_mfcc=13,
                hop_length=int(self.hop_size * self.sample_rate),
                n_fft=int(self.window_size * self.sample_rate)
            )
            
            # 델타 특징 추가
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 특징 결합
            features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
            
            return features.T  # (시간, 특징) 형태로 변환
            
        except ImportError:
            raise ModelError("librosa 라이브러리가 설치되지 않았습니다")
        except Exception as e:
            raise ModelError(f"특징 추출 실패: {str(e)}")
    
    def align_audio_text(self, audio_path: str, transcription_path: str) -> Dict:
        """오디오와 전사본을 정렬합니다.
        
        Args:
            audio_path (str): 오디오 파일 경로
            transcription_path (str): 전사본 파일 경로
        
        Returns:
            Dict: 정렬 결과
        
        Raises:
            AlignmentError: 정렬 실패 시
        """
        try:
            # 오디오 로드
            audio = self.load_audio(audio_path)
            
            # 전사본 로드
            transcription = self.load_transcription(transcription_path)
            
            # 특징 추출
            features = self.extract_features(audio)
            
            # 정렬 수행 (간단한 DTW 알고리즘 사용)
            alignment = self._perform_dtw_alignment(features, transcription)
            
            # 결과 정리
            result = {
                'audio_path': audio_path,
                'transcription': transcription,
                'alignment': alignment,
                'duration': len(audio) / self.sample_rate,
                'feature_shape': features.shape
            }
            
            # 결과 저장
            self.alignment_results[audio_path] = result
            
            return result
            
        except Exception as e:
            raise AlignmentError(f"정렬 실패: {str(e)}")
    
    def _perform_dtw_alignment(self, features: np.ndarray, transcription: str) -> List[Dict]:
        """DTW(Dynamic Time Warping)를 사용하여 정렬을 수행합니다.
        
        Args:
            features (np.ndarray): 오디오 특징
            transcription (str): 전사본 텍스트
        
        Returns:
            List[Dict]: 정렬된 시간 정보
        """
        # 간단한 DTW 구현 (실제로는 더 정교한 알고리즘 사용)
        n_frames = features.shape[0]
        n_chars = len(transcription)
        
        # DTW 행렬 초기화
        dtw_matrix = np.full((n_frames + 1, n_chars + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # 경로 추적을 위한 행렬
        path_matrix = np.zeros((n_frames + 1, n_chars + 1), dtype=int)
        
        # DTW 계산
        for i in range(1, n_frames + 1):
            for j in range(1, n_chars + 1):
                # 비용 계산 (간단한 유클리드 거리)
                cost = np.linalg.norm(features[i-1])
                
                # 이전 단계에서 최소 비용 찾기
                prev_costs = [
                    dtw_matrix[i-1, j],      # 삭제
                    dtw_matrix[i, j-1],      # 삽입
                    dtw_matrix[i-1, j-1]     # 대각선
                ]
                
                min_cost = min(prev_costs)
                min_idx = prev_costs.index(min_cost)
                
                dtw_matrix[i, j] = min_cost + cost
                path_matrix[i, j] = min_idx
        
        # 경로 역추적
        alignment = self._backtrack_path(path_matrix, transcription)
        
        return alignment
    
    def _backtrack_path(self, path_matrix: np.ndarray, transcription: str) -> List[Dict]:
        """DTW 경로를 역추적하여 정렬 결과를 생성합니다.
        
        Args:
            path_matrix (np.ndarray): 경로 행렬
            transcription (str): 전사본
        
        Returns:
            List[Dict]: 정렬된 시간 정보
        """
        alignment = []
        i, j = path_matrix.shape[0] - 1, path_matrix.shape[1] - 1
        
        while i > 0 and j > 0:
            char = transcription[j-1] if j > 0 else ''
            time = i / (self.sample_rate / (self.sample_rate * self.hop_size))
            
            alignment.append({
                'char': char,
                'start_time': max(0, time - self.hop_size),
                'end_time': time,
                'frame_index': i - 1
            })
            
            # 이전 단계로 이동
            prev_idx = path_matrix[i, j]
            if prev_idx == 0:      # 삭제
                i -= 1
            elif prev_idx == 1:    # 삽입
                j -= 1
            else:                   # 대각선
                i -= 1
                j -= 1
        
        # 시간 순서대로 정렬
        alignment.reverse()
        
        return alignment
    
    def save_alignment(self, output_path: str, audio_path: str = None):
        """정렬 결과를 파일로 저장합니다.
        
        Args:
            output_path (str): 출력 파일 경로
            audio_path (str): 특정 오디오 파일 경로 (None이면 모든 결과 저장)
        """
        try:
            if audio_path:
                results = {audio_path: self.alignment_results.get(audio_path, {})}
            else:
                results = self.alignment_results
            
            # JSON 형태로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            raise AlignmentError(f"결과 저장 실패: {str(e)}")
    
    def get_alignment_summary(self) -> Dict:
        """정렬 결과 요약을 반환합니다.
        
        Returns:
            Dict: 정렬 결과 요약
        """
        summary = {
            'total_files': len(self.alignment_results),
            'successful_alignments': 0,
            'failed_alignments': 0,
            'total_duration': 0.0,
            'average_alignment_quality': 0.0
        }
        
        for result in self.alignment_results.values():
            if 'alignment' in result:
                summary['successful_alignments'] += 1
                summary['total_duration'] += result.get('duration', 0.0)
            else:
                summary['failed_alignments'] += 1
        
        if summary['successful_alignments'] > 0:
            summary['average_alignment_quality'] = summary['successful_alignments'] / summary['total_files']
        
        return summary

def create_aligner(config: Dict) -> ForcedAligner:
    """강제 정렬기를 생성하는 편의 함수입니다.
    
    Args:
        config (Dict): 정렬 설정
    
    Returns:
        ForcedAligner: 강제 정렬기 인스턴스
    """
    return ForcedAligner(config)

if __name__ == "__main__":
    # 테스트
    config = {
        'model_path': 'models/alignment_model.pth',
        'sample_rate': 16000,
        'window_size': 0.025,
        'hop_size': 0.010,
        'min_silence_duration': 0.1
    }
    
    try:
        aligner = create_aligner(config)
        print("강제 정렬기 생성 성공")
        
        # 정렬 수행 예시 (실제 파일이 필요)
        # result = aligner.align_audio_text('audio.wav', 'transcript.txt')
        # print(f"정렬 결과: {result}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
