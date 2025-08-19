"""
IPA 기반 음성 인식 데이터셋
음성을 직접 IPA로 변환하고, IPA를 텍스트로 변환하는 방식
"""

import os
import json
import torch
import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from ..ipa.to_ipa import KoreanToIPA
from ..utils.io import load_json, ensure_dir

class IPADataset(Dataset):
    """IPA 기반 음성 인식 데이터셋"""
    
    def __init__(self, manifest_path: str, config: Dict, 
                 vocabulary_file: Optional[str] = None,
                 is_training: bool = True):
        """IPA 데이터셋을 초기화합니다.
        
        Args:
            manifest_path (str): 매니페스트 파일 경로
            config (Dict): 설정 딕셔너리
            vocabulary_file (str): IPA 어휘 파일 경로
            is_training (bool): 훈련 모드 여부
        """
        self.config = config
        self.is_training = is_training
        
        # 매니페스트 로드
        self.manifest = self._load_manifest(manifest_path)
        
        # IPA 변환기 초기화
        self.ipa_converter = KoreanToIPA()
        
        # 어휘 로드
        self.vocab = self._load_vocabulary(vocabulary_file)
        self.vocab_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_vocab = {idx: token for idx, token in enumerate(self.vocab)}
        
        # 오디오 설정
        self.sample_rate = config['data'].get('sample_rate', 16000)
        self.max_duration = config['data'].get('max_duration', 3.0)
        self.min_duration = config['data'].get('min_duration', 0.3)
        
        # 특징 추출 설정
        self.feature_config = config['data'].get('feature_type', 'mfcc')
        self.n_mfcc = config['data'].get('n_mfcc', 13)
        self.n_fft = config['data'].get('n_fft', 1024)
        self.hop_length = config['data'].get('hop_length', 256)
        
        # 데이터 증강 설정
        self.data_augmentation = config['training'].get('data_augmentation', {})
        self.aug_enabled = self.data_augmentation.get('enabled', False) and is_training
        
        print(f"IPA 데이터셋 초기화 완료: {len(self.manifest)}개 샘플")
        print(f"어휘 크기: {len(self.vocab)}")
        print(f"IPA 변환기: {type(self.ipa_converter).__name__}")
    
    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """매니페스트 파일을 로드합니다."""
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"매니페스트 파일이 존재하지 않습니다: {manifest_path}")
        
        manifest = load_json(manifest_path)
        
        # 리스트가 아닌 경우 처리
        if isinstance(manifest, dict):
            if 'train' in manifest and 'val' in manifest:
                # 전체 매니페스트에서 훈련/검증 분할
                return manifest['train'] if self.is_training else manifest['val']
            else:
                # 단일 매니페스트
                return [manifest]
        
        return manifest
    
    def _load_vocabulary(self, vocabulary_file: Optional[str]) -> List[str]:
        """IPA 어휘를 로드합니다."""
        if vocabulary_file and os.path.exists(vocabulary_file):
            # 사용자 정의 어휘 파일 사용
            with open(vocabulary_file, 'r', encoding='utf-8') as f:
                vocab = [line.strip() for line in f if line.strip()]
        else:
            # 기본 IPA 어휘 생성
            vocab = self._create_default_ipa_vocabulary()
        
        # 특수 토큰 추가
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        vocab = special_tokens + vocab
        
        return vocab
    
    def _create_default_ipa_vocabulary(self) -> List[str]:
        """기본 IPA 어휘를 생성합니다."""
        # 한국어 IPA 기본 음소들
        consonants = ['p', 't', 'k', 'tʃ', 's', 'h', 'm', 'n', 'ŋ', 'l', 'w', 'j']
        vowels = ['a', 'e', 'i', 'o', 'u', 'ɯ', 'ə', 'ɛ', 'ɔ', 'y', 'ø', 'ɨ']
        
        # 한국어 특유의 음소들
        korean_specific = ['ɕ', 'ʑ', 'tɕ', 'dʑ', 'ɭ', 'ɽ', 'ɸ', 'β', 'θ', 'ð']
        
        # 이중 자음
        double_consonants = ['pp', 'tt', 'kk', 'ss', 'tʃtʃ']
        
        # 음소 조합
        phoneme_combinations = []
        for c in consonants[:6]:  # 자음
            for v in vowels[:6]:  # 모음
                phoneme_combinations.append(c + v)
        
        # 전체 어휘
        vocab = consonants + vowels + korean_specific + double_consonants + phoneme_combinations
        
        # 중복 제거 및 정렬
        vocab = sorted(list(set(vocab)))
        
        return vocab
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터 샘플을 가져옵니다."""
        item = self.manifest[idx]
        
        # 오디오 로드 및 특징 추출
        audio_features = self._load_audio_features(item['audio_file'])
        
        # IPA 타겟 생성
        ipa_target = self._create_ipa_target(item['text'])
        
        # IPA를 인덱스로 변환
        ipa_indices = self._ipa_to_indices(ipa_target)
        
        return {
            'audio_features': audio_features,
            'ipa_target': ipa_target,
            'ipa_indices': torch.tensor(ipa_indices, dtype=torch.long),
            'text': item['text'],
            'audio_file': item['audio_file'],
            'duration': item.get('duration', 0.0)
        }
    
    def _load_audio_features(self, audio_file: str) -> torch.Tensor:
        """오디오 파일을 로드하고 특징을 추출합니다."""
        if not os.path.exists(audio_file):
            # 실제 오디오 파일이 없는 경우 더미 데이터 생성
            return self._create_dummy_audio_features()
        
        try:
            # 오디오 로드
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # 모노로 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 리샘플링
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # 길이 조정
            target_length = int(self.sample_rate * self.max_duration)
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            elif waveform.shape[1] < target_length:
                # 패딩
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # 특징 추출
            if self.feature_config == 'mfcc':
                features = self._extract_mfcc(waveform)
            elif self.feature_config == 'mel':
                features = self._extract_mel_spectrogram(waveform)
            else:
                features = self._extract_mfcc(waveform)  # 기본값
            
            # 데이터 증강 적용
            if self.aug_enabled:
                features = self._apply_data_augmentation(features)
            
            return features
            
        except Exception as e:
            print(f"오디오 로드 오류 ({audio_file}): {e}")
            return self._create_dummy_audio_features()
    
    def _create_dummy_audio_features(self) -> torch.Tensor:
        """더미 오디오 특징을 생성합니다 (테스트용)."""
        # MFCC 특징 크기 계산
        n_frames = int(self.max_duration * self.sample_rate / self.hop_length) + 1
        features = torch.randn(n_frames, self.n_mfcc)
        return features
    
    def _extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """MFCC 특징을 추출합니다."""
        # MFCC 변환
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        mfcc = mfcc_transform(waveform)
        return mfcc.squeeze(0)  # (n_mfcc, time)
    
    def _extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """멜 스펙트로그램을 추출합니다."""
        mel_transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=80
        )
        
        mel_spec = mel_transform(waveform)
        mel_spec_db = AmplitudeToDB()(mel_spec)
        return mel_spec_db.squeeze(0)  # (n_mels, time)
    
    def _apply_data_augmentation(self, features: torch.Tensor) -> torch.Tensor:
        """데이터 증강을 적용합니다."""
        if not self.aug_enabled:
            return features
        
        aug_config = self.data_augmentation
        
        # 노이즈 추가
        if aug_config.get('noise_level', 0) > 0:
            noise_level = aug_config['noise_level']
            noise = torch.randn_like(features) * noise_level
            features = features + noise
        
        # 시간 이동
        if aug_config.get('time_shift', 0) > 0:
            shift = int(features.shape[1] * aug_config['time_shift'])
            if shift > 0:
                features = torch.roll(features, shifts=shift, dims=1)
        
        # 피치 변화 (주파수 도메인에서)
        if aug_config.get('pitch_shift', 0) > 0:
            pitch_shift = aug_config['pitch_shift']
            # 간단한 피치 변화 (실제로는 더 복잡한 알고리즘 필요)
            features = features * (1 + pitch_shift * torch.randn_like(features))
        
        return features
    
    def _create_ipa_target(self, text: str) -> str:
        """텍스트를 IPA로 변환합니다."""
        try:
            # 한국어 텍스트를 IPA로 변환
            ipa = self.ipa_converter.text_to_ipa(text)
            return ipa
        except Exception as e:
            print(f"IPA 변환 오류 ({text}): {e}")
            # 오류 시 원본 텍스트 반환
            return text
    
    def _ipa_to_indices(self, ipa: str) -> List[int]:
        """IPA 문자열을 인덱스 리스트로 변환합니다."""
        indices = []
        
        # 시작 토큰
        indices.append(self.vocab_to_idx.get('<sos>', 2))
        
        # IPA 문자들을 인덱스로 변환
        for char in ipa:
            if char in self.vocab_to_idx:
                indices.append(self.vocab_to_idx[char])
            else:
                # 알 수 없는 문자는 <unk> 토큰으로
                indices.append(self.vocab_to_idx.get('<unk>', 1))
        
        # 끝 토큰
        indices.append(self.vocab_to_idx.get('<eos>', 3))
        
        return indices
    
    def get_vocab_size(self) -> int:
        """어휘 크기를 반환합니다."""
        return len(self.vocab)
    
    def get_vocab(self) -> List[str]:
        """어휘를 반환합니다."""
        return self.vocab
    
    def decode_indices(self, indices: List[int]) -> str:
        """인덱스 리스트를 IPA 문자열로 디코딩합니다."""
        ipa_chars = []
        
        for idx in indices:
            if idx in self.idx_to_vocab:
                token = self.idx_to_vocab[idx]
                if token not in ['<pad>', '<unk>', '<sos>', '<eos>']:
                    ipa_chars.append(token)
        
        return ''.join(ipa_chars)
    
    def ipa_to_text(self, ipa: str) -> str:
        """IPA를 한국어 텍스트로 변환합니다."""
        try:
            # IPA를 한국어로 역변환 (간단한 규칙 기반)
            # 실제로는 더 복잡한 역변환 로직이 필요
            text = self._simple_ipa_to_text(ipa)
            return text
        except Exception as e:
            print(f"IPA 역변환 오류 ({ipa}): {e}")
            return ipa
    
    def _simple_ipa_to_text(self, ipa: str) -> str:
        """간단한 IPA를 한국어 텍스트로 역변환합니다."""
        # 기본적인 IPA → 한국어 매핑
        ipa_to_korean = {
            'a': '아', 'e': '에', 'i': '이', 'o': '오', 'u': '우',
            'ɯ': '으', 'ə': '어', 'ɛ': '애', 'ɔ': '어', 'y': '위', 'ø': '외',
            'p': 'ㅂ', 't': 'ㄷ', 'k': 'ㄱ', 'tʃ': 'ㅊ', 's': 'ㅅ', 'h': 'ㅎ',
            'm': 'ㅁ', 'n': 'ㄴ', 'ŋ': 'ㅇ', 'l': 'ㄹ', 'w': 'ㅇ', 'j': 'ㅇ'
        }
        
        text = ''
        for char in ipa:
            if char in ipa_to_korean:
                text += ipa_to_korean[char]
            else:
                text += char
        
        return text

def create_ipa_dataloader(manifest_path: str, config: Dict, 
                         batch_size: int = 8, shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
    """IPA 데이터로더를 생성합니다."""
    dataset = IPADataset(manifest_path, config, is_training=shuffle)
    
    def collate_fn(batch):
        """배치 데이터를 정리하는 함수"""
        # 오디오 특징 길이 맞추기
        max_length = max(item['audio_features'].shape[1] for item in batch)
        
        audio_features = []
        ipa_indices = []
        texts = []
        audio_files = []
        
        for item in batch:
            # 오디오 특징 패딩
            features = item['audio_features']
            if features.shape[1] < max_length:
                padding = max_length - features.shape[1]
                features = torch.nn.functional.pad(features, (0, padding))
            audio_features.append(features)
            
            # IPA 인덱스
            ipa_indices.append(item['ipa_indices'])
            
            # 메타데이터
            texts.append(item['text'])
            audio_files.append(item['audio_file'])
        
        return {
            'audio_features': torch.stack(audio_features),
            'ipa_indices': torch.nn.utils.rnn.pad_sequence(ipa_indices, batch_first=True, padding_value=0),
            'texts': texts,
            'audio_files': audio_files
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=shuffle
    )

if __name__ == "__main__":
    # 테스트
    config = {
        'data': {
            'sample_rate': 16000,
            'max_duration': 3.0,
            'min_duration': 0.3,
            'feature_type': 'mfcc',
            'n_mfcc': 13,
            'n_fft': 1024,
            'hop_length': 256
        },
        'training': {
            'data_augmentation': {
                'enabled': True,
                'noise_level': 0.02,
                'time_shift': 0.05
            }
        }
    }
    
    # 더미 매니페스트 생성
    dummy_manifest = [
        {'audio_file': 'dummy1.wav', 'text': '바지'},
        {'audio_file': 'dummy2.wav', 'text': '가방'},
        {'audio_file': 'dummy3.wav', 'text': '접시'}
    ]
    
    # 더미 매니페스트 저장
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dummy_manifest, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    
    try:
        # 데이터셋 테스트
        dataset = IPADataset(temp_path, config)
        print(f"데이터셋 크기: {len(dataset)}")
        print(f"어휘 크기: {dataset.get_vocab_size()}")
        
        # 샘플 테스트
        sample = dataset[0]
        print(f"샘플 키: {sample.keys()}")
        print(f"오디오 특징 크기: {sample['audio_features'].shape}")
        print(f"IPA 타겟: {sample['ipa_target']}")
        print(f"IPA 인덱스: {sample['ipa_indices']}")
        
        # IPA 역변환 테스트
        decoded_ipa = dataset.decode_indices(sample['ipa_indices'].tolist())
        print(f"디코딩된 IPA: {decoded_ipa}")
        
        # IPA → 텍스트 변환 테스트
        text = dataset.ipa_to_text(decoded_ipa)
        print(f"IPA → 텍스트: {text}")
        
    finally:
        # 임시 파일 삭제
        os.unlink(temp_path) 