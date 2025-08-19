"""
단어 학습을 위한 모델 엔진
단어 수준의 음성 인식 학습에 최적화되어 있습니다.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config, Wav2Vec2Processor
from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperProcessor

from .base_engine import BaseEngine
from ..utils.io import save_torch_model, load_torch_model
from ..utils.seed import set_seed

class WordRecognitionModel(nn.Module):
    """단어 인식을 위한 신경망 모델"""
    
    def __init__(self, config: Dict):
        """단어 인식 모델을 초기화합니다.
        
        Args:
            config (Dict): 모델 설정
        """
        super().__init__()
        
        self.config = config
        self.model_type = config.get('type', 'wav2vec2')
        
        if self.model_type == 'wav2vec2':
            self._init_wav2vec2_model()
        elif self.model_type == 'whisper':
            self._init_whisper_model()
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
    
    def _init_wav2vec2_model(self):
        """Wav2Vec2 모델을 초기화합니다."""
        model_config = self.config.get('model', {})
        
        # 사전 훈련된 모델 로드
        pretrained_model = model_config.get('pretrained_model', 'facebook/wav2vec2-base')
        
        try:
            self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(pretrained_model)
            self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
            
            # 어휘 크기에 맞게 출력 레이어 조정
            vocab_size = model_config.get('vocab_size', 1000)
            if self.wav2vec2.config.vocab_size != vocab_size:
                self.wav2vec2.lm_head = nn.Linear(
                    self.wav2vec2.config.hidden_size, 
                    vocab_size
                )
                self.wav2vec2.config.vocab_size = vocab_size
            
            print(f"Wav2Vec2 모델 로드 완료: {pretrained_model}")
            
        except Exception as e:
            print(f"사전 훈련된 모델 로드 실패: {e}")
            # 기본 모델 생성
            self._create_default_wav2vec2()
    
    def _init_whisper_model(self):
        """Whisper 모델을 초기화합니다."""
        model_config = self.config.get('model', {})
        
        # 사전 훈련된 모델 로드
        pretrained_model = model_config.get('pretrained_model', 'openai/whisper-tiny')
        
        try:
            self.whisper = WhisperForConditionalGeneration.from_pretrained(pretrained_model)
            self.processor = WhisperProcessor.from_pretrained(pretrained_model)
            
            # 한국어 토크나이저 설정
            self.processor.tokenizer.set_prefix_tokens(language="ko", task="transcribe")
            
            print(f"Whisper 모델 로드 완료: {pretrained_model}")
            
        except Exception as e:
            print(f"사전 훈련된 모델 로드 실패: {e}")
            # 기본 모델 생성
            self._create_default_whisper()
    
    def _create_default_wav2vec2(self):
        """기본 Wav2Vec2 모델을 생성합니다."""
        model_config = self.config.get('model', {})
        
        config = Wav2Vec2Config(
            hidden_size=model_config.get('hidden_size', 768),
            num_hidden_layers=model_config.get('num_hidden_layers', 12),
            num_attention_heads=model_config.get('num_attention_heads', 12),
            intermediate_size=model_config.get('intermediate_size', 3072),
            vocab_size=model_config.get('vocab_size', 1000),
            dropout=model_config.get('dropout', 0.1),
            attention_dropout=model_config.get('attention_dropout', 0.1)
        )
        
        self.wav2vec2 = Wav2Vec2ForCTC(config)
        print("기본 Wav2Vec2 모델 생성 완료")
    
    def _create_default_whisper(self):
        """기본 Whisper 모델을 생성합니다."""
        model_config = self.config.get('model', {})
        
        config = WhisperConfig(
            vocab_size=model_config.get('vocab_size', 1000),
            num_mel_bins=80,
            encoder_layers=model_config.get('num_hidden_layers', 6),
            encoder_attention_heads=model_config.get('num_attention_heads', 6),
            decoder_layers=model_config.get('num_hidden_layers', 6),
            decoder_attention_heads=model_config.get('num_attention_heads', 6),
            d_model=model_config.get('hidden_size', 384)
        )
        
        self.whisper = WhisperForConditionalGeneration(config)
        print("기본 Whisper 모델 생성 완료")
    
    def forward(self, input_values, attention_mask=None, labels=None):
        """순전파를 수행합니다."""
        if self.model_type == 'wav2vec2':
            return self._forward_wav2vec2(input_values, attention_mask, labels)
        elif self.model_type == 'whisper':
            return self._forward_whisper(input_values, attention_mask, labels)
    
    def _forward_wav2vec2(self, input_values, attention_mask=None, labels=None):
        """Wav2Vec2 순전파"""
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def _forward_whisper(self, input_features, attention_mask=None, labels=None):
        """Whisper 순전파"""
        outputs = self.whisper(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self, input_values, max_length=50):
        """텍스트를 생성합니다."""
        if self.model_type == 'wav2vec2':
            return self._generate_wav2vec2(input_values, max_length)
        elif self.model_type == 'whisper':
            return self._generate_whisper(input_values, max_length)
    
    def _generate_wav2vec2(self, input_values, max_length):
        """Wav2Vec2 텍스트 생성"""
        with torch.no_grad():
            logits = self.wav2vec2(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            return predicted_ids
    
    def _generate_whisper(self, input_features, max_length):
        """Whisper 텍스트 생성"""
        with torch.no_grad():
            generated_ids = self.whisper.generate(
                input_features,
                max_length=max_length,
                language="ko",
                task="transcribe"
            )
            return generated_ids

class WordEngine(BaseEngine):
    """단어 학습을 위한 엔진"""
    
    def __init__(self, config: Dict):
        """단어 엔진을 초기화합니다.
        
        Args:
            config (Dict): 엔진 설정
        """
        super().__init__(config)
        
        # 시드 설정
        set_seed(config.get('seed', 42))
        
        # 모델 생성
        self.model = WordRecognitionModel(config)
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 옵티마이저 설정
        self._setup_optimizer()
        
        # 스케줄러 설정
        self._setup_scheduler()
        
        print(f"단어 엔진 초기화 완료 (디바이스: {self.device})")
    
    def _setup_optimizer(self):
        """옵티마이저를 설정합니다."""
        training_config = self.config.get('training', {})
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    
    def _setup_scheduler(self):
        """스케줄러를 설정합니다."""
        training_config = self.config.get('training', {})
        scheduler_type = training_config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('max_steps', 50000)
            )
        elif scheduler_type == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=training_config.get('max_steps', 50000)
            )
        else:
            self.scheduler = None
    
    def train_step(self, batch: Dict) -> Dict:
        """한 단계의 훈련을 수행합니다."""
        self.model.train()
        
        # 데이터를 디바이스로 이동
        audio_features = batch['audio_features'].to(self.device)
        text_tokens = batch['text_tokens'].to(self.device)
        
        # 순전파
        outputs = self.model(audio_features, labels=text_tokens)
        loss = outputs.loss
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑
        training_config = self.config.get('training', {})
        max_grad_norm = training_config.get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate_step(self, batch: Dict) -> Dict:
        """한 단계의 평가를 수행합니다."""
        self.model.eval()
        
        with torch.no_grad():
            audio_features = batch['audio_features'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            
            # 순전파
            outputs = self.model(audio_features, labels=text_tokens)
            loss = outputs.loss
            
            # 생성
            generated_ids = self.model.generate(audio_features)
            
            # 디코딩
            predicted_texts = []
            for ids in generated_ids:
                if self.model.model_type == 'wav2vec2':
                    # Wav2Vec2는 CTC 출력을 사용
                    predicted_text = self._decode_ctc(ids)
                else:
                    # Whisper는 시퀀스 출력을 사용
                    predicted_text = self._decode_sequence(ids)
                predicted_texts.append(predicted_text)
            
            return {
                'loss': loss.item(),
                'predicted_texts': predicted_texts,
                'reference_texts': batch['texts']
            }
    
    def _decode_ctc(self, ids: torch.Tensor) -> str:
        """CTC 출력을 디코딩합니다."""
        # 연속된 같은 ID 제거
        unique_ids = []
        prev_id = None
        for id_val in ids:
            if id_val != prev_id and id_val != 0:  # 0은 <pad> 토큰
                unique_ids.append(id_val.item())
            prev_id = id_val
        
        # 어휘를 사용하여 텍스트로 변환
        if hasattr(self, 'vocab'):
            idx_to_word = {v: k for k, v in self.vocab.items()}
            words = [idx_to_word.get(idx, '<unk>') for idx in unique_ids]
            return ' '.join(words)
        
        return str(unique_ids)
    
    def _decode_sequence(self, ids: torch.Tensor) -> str:
        """시퀀스 출력을 디코딩합니다."""
        # 특수 토큰 제거
        valid_ids = [id_val.item() for id_val in ids if id_val not in [0, 2, 3]]  # <pad>, <sos>, <eos>
        
        # 어휘를 사용하여 텍스트로 변환
        if hasattr(self, 'vocab'):
            idx_to_word = {v: k for k, v in self.vocab.items()}
            words = [idx_to_word.get(idx, '<unk>') for idx in valid_ids]
            return ' '.join(words)
        
        return str(valid_ids)
    
    def save_model(self, path: str):
        """모델을 저장합니다."""
        save_torch_model(self.model, path)
        print(f"모델이 저장되었습니다: {path}")
    
    def load_model(self, path: str):
        """모델을 로드합니다."""
        load_torch_model(self.model, path)
        print(f"모델이 로드되었습니다: {path}")
    
    def set_vocab(self, vocab: Dict[str, int]):
        """어휘를 설정합니다."""
        self.vocab = vocab
        print(f"어휘가 설정되었습니다: {len(vocab)}개 토큰")

def create_word_engine(config: Dict) -> WordEngine:
    """단어 엔진을 생성하는 편의 함수입니다.
    
    Args:
        config (Dict): 엔진 설정
    
    Returns:
        WordEngine: 단어 엔진 인스턴스
    """
    return WordEngine(config)

if __name__ == "__main__":
    # 테스트
    config = {
        'model': {
            'type': 'wav2vec2',
            'pretrained_model': 'facebook/wav2vec2-base',
            'vocab_size': 1000
        },
        'training': {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'max_steps': 50000
        },
        'seed': 42
    }
    
    try:
        engine = create_word_engine(config)
        print("단어 엔진 생성 성공")
        
        # 모델 정보 출력
        total_params = sum(p.numel() for p in engine.model.parameters())
        print(f"총 파라미터 수: {total_params:,}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}") 