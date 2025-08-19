"""
데이터 전처리를 위한 모듈
오디오 파일과 전사본을 학습에 적합한 형태로 변환합니다.
"""

import os
import json
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import torchaudio

from ..utils.io import ensure_dir, save_json, load_json
from ..ipa.to_ipa import text_to_ipa

class DataPreprocessor:
    """데이터 전처리를 수행하는 클래스"""
    
    def __init__(self, config: Dict):
        """데이터 전처리기를 초기화합니다.
        
        Args:
            config (Dict): 전처리 설정
        """
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.max_duration = config.get('max_duration', 30.0)
        self.min_duration = config.get('min_duration', 1.0)
        self.normalize_audio = config.get('normalize_audio', True)
        self.add_noise = config.get('add_noise', False)
        self.noise_level = config.get('noise_level', 0.01)
        
    def preprocess_audio(self, audio_path: str, output_path: str = None) -> Dict:
        """오디오 파일을 전처리합니다.
        
        Args:
            audio_path (str): 입력 오디오 파일 경로
            output_path (str): 출력 파일 경로 (None이면 입력 파일과 동일한 디렉토리에 저장)
        
        Returns:
            Dict: 전처리 결과 정보
        """
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 길이 확인
            duration = len(audio) / self.sample_rate
            
            if duration < self.min_duration:
                raise ValueError(f"오디오가 너무 짧습니다: {duration:.2f}초")
            
            if duration > self.max_duration:
                # 긴 오디오는 자르기
                max_samples = int(self.max_duration * self.sample_rate)
                audio = audio[:max_samples]
                duration = self.max_duration
                print(f"오디오가 잘렸습니다: {duration:.2f}초")
            
            # 오디오 정규화
            if self.normalize_audio:
                audio = librosa.util.normalize(audio)
            
            # 노이즈 추가 (선택사항)
            if self.add_noise:
                noise = np.random.normal(0, self.noise_level, len(audio))
                audio = audio + noise
                audio = np.clip(audio, -1.0, 1.0)
            
            # 출력 경로 설정
            if output_path is None:
                input_path = Path(audio_path)
                output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
            
            # 전처리된 오디오 저장
            ensure_dir(os.path.dirname(output_path))
            librosa.output.write_wav(output_path, audio, self.sample_rate)
            
            # 결과 정보 반환
            result = {
                'input_path': audio_path,
                'output_path': str(output_path),
                'original_duration': duration,
                'sample_rate': self.sample_rate,
                'normalized': self.normalize_audio,
                'noise_added': self.add_noise
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"오디오 전처리 실패: {str(e)}")
    
    def preprocess_transcript(self, transcript: str, convert_to_ipa: bool = True) -> Dict:
        """전사본을 전처리합니다.
        
        Args:
            transcript (str): 원본 전사본
            convert_to_ipa (bool): IPA 변환 여부
        
        Returns:
            Dict: 전처리 결과
        """
        # 텍스트 정규화
        normalized_text = transcript.strip()
        
        # 특수문자 처리
        normalized_text = normalized_text.replace('"', '').replace("'", '')
        normalized_text = ' '.join(normalized_text.split())  # 연속 공백 제거
        
        result = {
            'original': transcript,
            'normalized': normalized_text,
            'length': len(normalized_text),
            'word_count': len(normalized_text.split())
        }
        
        # IPA 변환
        if convert_to_ipa:
            try:
                ipa_text = text_to_ipa(normalized_text)
                result['ipa'] = ipa_text
                result['ipa_length'] = len(ipa_text)
            except Exception as e:
                print(f"IPA 변환 실패: {e}")
                result['ipa'] = None
                result['ipa_length'] = 0
        
        return result
    
    def create_audio_features(self, audio_path: str) -> np.ndarray:
        """오디오에서 특징을 추출합니다.
        
        Args:
            audio_path (str): 오디오 파일 경로
        
        Returns:
            np.ndarray: 특징 벡터
        """
        # 오디오 로드
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # MFCC 특징 추출
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr,
            n_mfcc=13,
            hop_length=512,
            n_fft=2048
        )
        
        # 델타 특징
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 특징 결합
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        
        return features.T  # (시간, 특징) 형태
    
    def batch_preprocess(self, manifest_path: str, output_dir: str) -> Dict:
        """매니페스트 파일을 기반으로 배치 전처리를 수행합니다.
        
        Args:
            manifest_path (str): 매니페스트 파일 경로
            output_dir (str): 출력 디렉토리
        
        Returns:
            Dict: 배치 전처리 결과
        """
        # 매니페스트 로드
        manifest = load_json(manifest_path)
        
        # 출력 디렉토리 생성
        ensure_dir(output_dir)
        processed_manifest = []
        
        total_files = len(manifest.get('train', [])) + len(manifest.get('val', []))
        processed_count = 0
        
        # 훈련 데이터 전처리
        for split_name in ['train', 'val']:
            split_data = manifest.get(split_name, [])
            
            for item in split_data:
                try:
                    # 오디오 전처리
                    audio_output = os.path.join(
                        output_dir, 
                        f"{split_name}_{Path(item['audio_file']).stem}_processed.wav"
                    )
                    
                    audio_result = self.preprocess_audio(
                        item['audio_file'], 
                        audio_output
                    )
                    
                    # 전사본 전처리
                    transcript_result = self.preprocess_transcript(
                        item['text'],
                        convert_to_ipa=True
                    )
                    
                    # 전처리된 항목 생성
                    processed_item = {
                        'audio_file': audio_result['output_path'],
                        'transcript_file': item['transcript_file'],
                        'text': transcript_result['normalized'],
                        'ipa': transcript_result.get('ipa'),
                        'duration': audio_result['original_duration'],
                        'original_audio': item['audio_file'],
                        'preprocessing_info': {
                            'audio': audio_result,
                            'transcript': transcript_result
                        }
                    }
                    
                    processed_manifest.append(processed_item)
                    processed_count += 1
                    
                    print(f"진행률: {processed_count}/{total_files} - {Path(item['audio_file']).name}")
                    
                except Exception as e:
                    print(f"전처리 실패 ({item['audio_file']}): {e}")
                    continue
        
        # 전처리된 매니페스트 저장
        output_manifest_path = os.path.join(output_dir, 'processed_manifest.json')
        save_json(processed_manifest, output_manifest_path)
        
        # 결과 요약
        summary = {
            'total_files': total_files,
            'processed_files': len(processed_manifest),
            'failed_files': total_files - len(processed_manifest),
            'output_directory': output_dir,
            'manifest_path': output_manifest_path
        }
        
        return summary
    
    def validate_preprocessed_data(self, manifest_path: str) -> Dict:
        """전처리된 데이터의 유효성을 검사합니다.
        
        Args:
            manifest_path (str): 전처리된 매니페스트 파일 경로
        
        Returns:
            Dict: 검증 결과
        """
        try:
            manifest = load_json(manifest_path)
            
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'file_count': len(manifest),
                'total_duration': 0.0,
                'ipa_coverage': 0
            }
            
            for item in manifest:
                # 오디오 파일 확인
                if not os.path.exists(item['audio_file']):
                    validation_result['errors'].append(
                        f"오디오 파일이 존재하지 않습니다: {item['audio_file']}"
                    )
                
                # 전사본 확인
                if not item.get('text', '').strip():
                    validation_result['warnings'].append(
                        f"빈 전사본: {item['audio_file']}"
                    )
                
                # IPA 확인
                if item.get('ipa'):
                    validation_result['ipa_coverage'] += 1
                
                # 길이 정보 누적
                validation_result['total_duration'] += item.get('duration', 0.0)
            
            # 오류가 있으면 유효하지 않음
            if validation_result['errors']:
                validation_result['is_valid'] = False
            
            # IPA 커버리지 계산
            if validation_result['file_count'] > 0:
                validation_result['ipa_coverage'] /= validation_result['file_count']
            
            return validation_result
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f"검증 실패: {str(e)}"],
                'warnings': [],
                'file_count': 0,
                'total_duration': 0.0,
                'ipa_coverage': 0.0
            }

def create_preprocessor(config: Dict) -> DataPreprocessor:
    """데이터 전처리기를 생성하는 편의 함수입니다.
    
    Args:
        config (Dict): 전처리 설정
    
    Returns:
        DataPreprocessor: 데이터 전처리기 인스턴스
    """
    return DataPreprocessor(config)

if __name__ == "__main__":
    # 테스트
    config = {
        'sample_rate': 16000,
        'max_duration': 30.0,
        'min_duration': 1.0,
        'normalize_audio': True,
        'add_noise': False,
        'noise_level': 0.01
    }
    
    try:
        preprocessor = create_preprocessor(config)
        print("데이터 전처리기 생성 성공")
        
        # 전사본 전처리 테스트
        test_text = "안녕하세요, 오늘 날씨가 좋네요."
        result = preprocessor.preprocess_transcript(test_text)
        print(f"전사본 전처리 결과: {result}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
