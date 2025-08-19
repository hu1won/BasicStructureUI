"""
단어 학습을 위한 데이터셋 클래스
단어 수준의 음성 인식 학습에 최적화되어 있습니다.
"""

import os
import json
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import librosa
from pathlib import Path

from ..utils.io import load_json
from ..ipa.to_ipa import text_to_ipa

class WordDataset(Dataset):
    """단어 학습을 위한 데이터셋 클래스"""
    
    def __init__(self, manifest_path: str, config: Dict, tokenizer=None):
        """단어 데이터셋을 초기화합니다.
        
        Args:
            manifest_path (str): 매니페스트 파일 경로
            config (Dict): 데이터셋 설정
            tokenizer: 텍스트 토크나이저
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # 매니페스트 로드
        self.manifest = self._load_manifest(manifest_path)
        
        # 오디오 설정
        self.sample_rate = config.get('sample_rate', 16000)
        self.max_duration = config.get('max_duration', 5.0)
        self.min_duration = config.get('min_duration', 0.5)
        
        # 특징 추출 설정
        self.feature_type = config.get('feature_type', 'mfcc')
        self.n_mfcc = config.get('n_mfcc', 13)
        self.n_fft = config.get('n_fft', 2048)
        self.hop_length = config.get('hop_length', 512)
        
        # IPA 변환 설정
        self.ipa_enabled = config.get('ipa', {}).get('enabled', True)
        self.apply_post_rules = config.get('ipa', {}).get('apply_post_rules', True)
        
        # 어휘 설정
        self.vocab = self._load_vocabulary()
        
        print(f"데이터셋 로드 완료: {len(self.manifest)}개 샘플")
    
    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """매니페스트 파일을 로드합니다."""
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"매니페스트 파일이 존재하지 않습니다: {manifest_path}")
        
        manifest = load_json(manifest_path)
        
        # 단일 리스트인 경우 처리
        if isinstance(manifest, list):
            return manifest
        elif isinstance(manifest, dict) and 'train' in manifest:
            # 훈련/검증 분할된 경우
            return manifest.get('train', []) + manifest.get('val', [])
        else:
            raise ValueError("올바르지 않은 매니페스트 형식입니다.")
    
    def _load_vocabulary(self) -> Dict[str, int]:
        """어휘 파일을 로드합니다."""
        vocab_path = self.config.get('vocabulary_file')
        if not vocab_path or not os.path.exists(vocab_path):
            # 기본 어휘 생성
            return self._create_default_vocabulary()
        
        vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word = line.strip()
                if word:
                    vocab[word] = i
        
        # 특수 토큰 추가
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for token in special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        return vocab
    
    def _create_default_vocabulary(self) -> Dict[str, int]:
        """기본 어휘를 생성합니다."""
        # 매니페스트에서 모든 단어 수집
        all_words = set()
        for item in self.manifest:
            text = item.get('text', '')
            words = text.split()
            all_words.update(words)
        
        # 특수 토큰 추가
        vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        
        # 단어 추가
        for i, word in enumerate(sorted(all_words)):
            vocab[word] = i + 4
        
        print(f"기본 어휘 생성: {len(vocab)}개 토큰")
        return vocab
    
    def __len__(self) -> int:
        """데이터셋 크기를 반환합니다."""
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터셋에서 하나의 샘플을 가져옵니다."""
        item = self.manifest[idx]
        
        # 오디오 로드 및 전처리
        audio_features = self._load_audio_features(item['audio_file'])
        
        # 텍스트 처리
        text = item.get('text', '')
        text_tokens = self._process_text(text)
        
        # IPA 변환
        ipa_text = None
        if self.ipa_enabled:
            try:
                ipa_text = text_to_ipa(text, apply_post_rules=self.apply_post_rules)
            except Exception as e:
                print(f"IPA 변환 실패: {e}")
                ipa_text = text
        
        return {
            'audio_features': audio_features,
            'text': text,
            'text_tokens': text_tokens,
            'ipa_text': ipa_text or text,
            'audio_path': item['audio_file'],
            'duration': item.get('duration', 0.0)
        }
    
    def _load_audio_features(self, audio_path: str) -> torch.Tensor:
        """오디오 파일을 로드하고 특징을 추출합니다."""
        try:
            # 오디오 로드
            audio, sr = torchaudio.load(audio_path)
            
            # 모노로 변환
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # 샘플 레이트 변환
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # 길이 확인 및 조정
            max_samples = int(self.max_duration * self.sample_rate)
            if audio.shape[1] > max_samples:
                audio = audio[:, :max_samples]
            elif audio.shape[1] < int(self.min_duration * self.sample_rate):
                # 짧은 오디오는 패딩
                padding = max_samples - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            
            # 특징 추출
            if self.feature_type == 'mfcc':
                features = self._extract_mfcc(audio.squeeze())
            else:
                # 기본 특징 (오디오 자체)
                features = audio.squeeze()
            
            return features
            
        except Exception as e:
            print(f"오디오 로드 실패 ({audio_path}): {e}")
            # 빈 특징 반환
            return torch.zeros(self.n_mfcc, 100)
    
    def _extract_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        """MFCC 특징을 추출합니다."""
        # PyTorch 기반 MFCC 추출
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc_transform(audio)
    
    def _process_text(self, text: str) -> torch.Tensor:
        """텍스트를 토큰으로 변환합니다."""
        if not text:
            return torch.tensor([self.vocab['<unk>']])
        
        words = text.split()
        tokens = [self.vocab.get(word, self.vocab['<unk>']) for word in words]
        
        # 시작/끝 토큰 추가
        tokens = [self.vocab['<sos>']] + tokens + [self.vocab['<eos>']]
        
        return torch.tensor(tokens)
    
    def get_vocab_size(self) -> int:
        """어휘 크기를 반환합니다."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """어휘를 반환합니다."""
        return self.vocab.copy()
    
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """토큰을 텍스트로 디코딩합니다."""
        if not hasattr(self, '_idx_to_word'):
            self._idx_to_word = {v: k for k, v in self.vocab.items()}
        
        words = []
        for token in tokens:
            if token.item() in self._idx_to_word:
                word = self._idx_to_word[token.item()]
                if word not in ['<pad>', '<sos>', '<eos>']:
                    words.append(word)
        
        return ' '.join(words)

def create_word_dataloader(manifest_path: str, config: Dict, 
                          batch_size: int = 32, shuffle: bool = True,
                          num_workers: int = 4) -> DataLoader:
    """단어 데이터로더를 생성합니다.
    
    Args:
        manifest_path (str): 매니페스트 파일 경로
        config (Dict): 데이터셋 설정
        batch_size (int): 배치 크기
        shuffle (bool): 셔플 여부
        num_workers (int): 워커 수
    
    Returns:
        DataLoader: 단어 데이터로더
    """
    dataset = WordDataset(manifest_path, config)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """배치 데이터를 정렬합니다."""
    # 오디오 특징 정렬
    audio_features = [item['audio_features'] for item in batch]
    audio_lengths = [len(feat) for feat in audio_features]
    max_audio_length = max(audio_lengths)
    
    # 패딩된 오디오 특징
    padded_audio = []
    for feat in audio_features:
        if len(feat) < max_audio_length:
            padding = max_audio_length - len(feat)
            if feat.dim() == 1:
                padded = torch.nn.functional.pad(feat, (0, padding))
            else:
                padded = torch.nn.functional.pad(feat, (0, 0, 0, padding))
        else:
            padded = feat
        padded_audio.append(padded)
    
    # 텍스트 토큰 정렬
    text_tokens = [item['text_tokens'] for item in batch]
    text_lengths = [len(tokens) for tokens in text_tokens]
    max_text_length = max(text_lengths)
    
    # 패딩된 텍스트 토큰
    padded_text = []
    for tokens in text_tokens:
        if len(tokens) < max_text_length:
            padding = max_text_length - len(tokens)
            padded = torch.nn.functional.pad(tokens, (0, padding), value=0)  # <pad> 토큰
        else:
            padded = tokens
        padded_text.append(padded)
    
    return {
        'audio_features': torch.stack(padded_audio),
        'audio_lengths': torch.tensor(audio_lengths),
        'text_tokens': torch.stack(padded_text),
        'text_lengths': torch.tensor(text_lengths),
        'texts': [item['text'] for item in batch],
        'ipa_texts': [item['ipa_text'] for item in batch],
        'audio_paths': [item['audio_path'] for item in batch],
        'durations': torch.tensor([item['duration'] for item in batch])
    }

if __name__ == "__main__":
    # 테스트
    config = {
        'sample_rate': 16000,
        'max_duration': 5.0,
        'min_duration': 0.5,
        'feature_type': 'mfcc',
        'n_mfcc': 13,
        'n_fft': 2048,
        'hop_length': 512,
        'ipa': {'enabled': True, 'apply_post_rules': True}
    }
    
    try:
        # 데이터셋 테스트 (실제 매니페스트 파일이 필요)
        # dataset = WordDataset('data/manifest_word_train.json', config)
        # print(f"데이터셋 크기: {len(dataset)}")
        # print(f"어휘 크기: {dataset.get_vocab_size()}")
        
        print("단어 데이터셋 클래스 생성 성공")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}") 