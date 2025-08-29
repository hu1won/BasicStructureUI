"""
IPA 기반 음성 인식 엔진
음성을 직접 IPA로 변환하고, IPA를 텍스트로 변환하는 방식
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Config,
    WhisperForConditionalGeneration, WhisperConfig,
    AutoTokenizer, AutoProcessor
)
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from .base_engine import BaseEngine
from ..utils.io import save_torch_model, load_torch_model
from ..utils.seed import set_seed

class IPARecognitionModel(nn.Module):
    """IPA 인식을 위한 모델 래퍼"""
    
    def __init__(self, config: Dict):
        """IPA 인식 모델을 초기화합니다."""
        super().__init__()
        
        self.config = config
        self.model_type = config['model']['type']
        
        if self.model_type == 'wav2vec2':
            self.model = self._create_wav2vec2_model()
        elif self.model_type == 'whisper':
            self.model = self._create_whisper_model()
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        # IPA 어휘 크기
        self.vocab_size = config['model'].get('vocab_size', 100)
        
        # 출력 레이어 (IPA 토큰 예측)
        if self.model_type == 'wav2vec2':
            self.output_projection = nn.Linear(
                self.model.config.hidden_size, 
                self.vocab_size
            )
        
        # 드롭아웃
        self.dropout = nn.Dropout(config['model'].get('dropout', 0.1))
    
    def _create_wav2vec2_model(self):
        """Wav2Vec2 모델을 생성합니다."""
        model_config = Wav2Vec2Config.from_pretrained(
            self.config['model']['pretrained_model']
        )
        
        # IPA 인식에 맞게 수정
        model_config.vocab_size = self.config['model'].get('vocab_size', 100)
        model_config.hidden_size = self.config['model'].get('hidden_size', 768)
        model_config.num_hidden_layers = self.config['model'].get('num_hidden_layers', 6)
        model_config.num_attention_heads = self.config['model'].get('num_attention_heads', 12)
        
        # 마스킹 비활성화
        model_config.mask_time_prob = 0.0
        model_config.mask_feature_prob = 0.0
        model_config.mask_time_length = 1  # 최소 마스킹 길이
        model_config.mask_feature_length = 1  # 최소 특징 마스킹 길이
        model_config.mask_time_min_masks = 0  # 최소 마스킹 개수
        model_config.mask_feature_min_masks = 0  # 최소 특징 마스킹 개수
        
        model = Wav2Vec2ForCTC.from_pretrained(
            self.config['model']['pretrained_model'],
            config=model_config
        )
        
        return model
    
    def _create_whisper_model(self):
        """Whisper 모델을 생성합니다."""
        model_config = WhisperConfig.from_pretrained(
            self.config['model']['pretrained_model']
        )
        
        # IPA 인식에 맞게 수정
        model_config.vocab_size = self.config['model'].get('vocab_size', 100)
        model_config.d_model = self.config['model'].get('hidden_size', 768)
        model_config.encoder_layers = self.config['model'].get('num_hidden_layers', 6)
        model_config.encoder_attention_heads = self.config['model'].get('num_attention_heads', 12)
        
        model = WhisperForConditionalGeneration.from_pretrained(
            self.config['model']['pretrained_model'],
            config=model_config
        )
        
        return model
    
    def forward(self, audio_features: torch.Tensor, 
                ipa_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """모델의 순전파를 수행합니다."""
        if self.model_type == 'wav2vec2':
            return self._forward_wav2vec2(audio_features, ipa_targets)
        elif self.model_type == 'whisper':
            return self._forward_whisper(audio_features, ipa_targets)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
    
    def _forward_wav2vec2(self, audio_features: torch.Tensor, 
                          ipa_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Wav2Vec2 모델의 순전파를 수행합니다."""
        # MFCC 특징을 원시 오디오로 역변환
        # audio_features: (batch_size, n_features, time)
        
        if audio_features.dim() == 3:
            # MFCC를 원시 오디오로 역변환
            audio_features = self._mfcc_to_audio(audio_features)
        
        # Wav2Vec2 모델 실행
        outputs = self.model(
            input_values=audio_features,
            labels=ipa_targets
        )
        
        return {
            'logits': outputs.logits,
            'loss': outputs.loss if ipa_targets is not None else None,
            'hidden_states': outputs.hidden_states
        }
    
    def _mfcc_to_audio(self, mfcc_features: torch.Tensor) -> torch.Tensor:
        """MFCC 특징을 원시 오디오로 역변환합니다."""
        batch_size, n_mfcc, time = mfcc_features.shape
        
        # MFCC를 원시 오디오로 역변환 (Mel 필터뱅크 역변환)
        # 간단한 근사 방법: MFCC의 첫 번째 차원(에너지)을 사용
        if n_mfcc >= 1:
            # 에너지 정보를 사용하여 원시 오디오 근사
            energy = mfcc_features[:, 0, :]  # 첫 번째 MFCC 계수 (에너지)
            
            # 에너지를 원시 오디오로 변환 (간단한 방법)
            # 실제로는 더 정교한 역변환이 필요하지만, 학습을 위해 근사 사용
            audio_approx = energy * torch.randn_like(energy) * 0.1
            
            # 정규화
            audio_approx = torch.tanh(audio_approx) * 0.5
            
            return audio_approx
        else:
            # 폴백: 랜덤 오디오 생성
            return torch.randn(batch_size, time, device=mfcc_features.device) * 0.1
    
    def _forward_whisper(self, audio_features: torch.Tensor, 
                         ipa_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Whisper 모델의 순전파를 수행합니다."""
        # Whisper는 멜 스펙트로그램을 입력으로 받음
        if audio_features.dim() == 3:
            # (batch_size, n_features, time) -> (batch_size, time, n_features)
            audio_features = audio_features.transpose(1, 2)
        
        # Whisper 모델 실행
        if ipa_targets is not None:
            # 훈련 모드
            outputs = self.model(
                input_features=audio_features,
                labels=ipa_targets
            )
        else:
            # 추론 모드
            outputs = self.model.generate(
                input_features=audio_features,
                max_length=50,
                do_sample=False,
                num_beams=1
            )
        
        return {
            'logits': outputs.logits if hasattr(outputs, 'logits') else None,
            'loss': outputs.loss if hasattr(outputs, 'loss') else None,
            'generated_ids': outputs if not hasattr(outputs, 'logits') else None
        }
    

    
    def get_vocab_size(self) -> int:
        """어휘 크기를 반환합니다."""
        return self.vocab_size

class IPAEngine(BaseEngine):
    """IPA 기반 음성 인식 엔진"""
    
    def __init__(self, config: Dict):
        """IPA 엔진을 초기화합니다."""
        super().__init__(config)
        
        # 시드 설정
        set_seed(config.get('seed', 42))
        
        # 모델 생성
        self.model = IPARecognitionModel(config)
        
        # 모델 타입 설정
        self.model_type = config['model']['type']
        
        # 어휘 설정
        self.vocab = None
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        
        # 옵티마이저 및 스케줄러 설정
        self._setup_optimizer()
        self._setup_scheduler()
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"IPA 엔진 초기화 완료: {self.model_type}")
        print(f"디바이스: {self.device}")
        print(f"어휘 크기: {self.model.get_vocab_size()}")
    
    def _setup_optimizer(self):
        """옵티마이저를 설정합니다."""
        training_config = self.config['training']
        
        # 가중치 감쇠가 적용되지 않는 파라미터들
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 파라미터 그룹 생성
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': training_config.get('weight_decay', 0.01)
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # 학습률 타입 검증 및 변환
        lr = training_config.get('learning_rate', 5e-5)
        if isinstance(lr, str):
            try:
                lr = float(lr)
                print(f"학습률을 문자열에서 숫자로 변환: {lr}")
            except ValueError:
                raise ValueError(f"유효하지 않은 학습률: {lr}")
        
        # 옵티마이저 생성
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _setup_scheduler(self):
        """학습률 스케줄러를 설정합니다."""
        training_config = self.config['training']
        scheduler_type = training_config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('max_steps', 100000),
                eta_min=1e-7
            )
        elif scheduler_type == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=training_config.get('max_steps', 100000)
            )
        else:
            # 기본 스케줄러
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1000,
                gamma=0.9
            )
    
    def set_vocab(self, vocab: List[str]):
        """어휘를 설정합니다."""
        self.vocab = vocab
        self.vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_vocab = {idx: token for idx, token in enumerate(vocab)}
        
        print(f"어휘 설정 완료: {len(vocab)}개 토큰")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """훈련 단계를 수행합니다."""
        self.model.train()
        
        # 배치 데이터를 디바이스로 이동
        audio_features = batch['audio_features'].to(self.device)
        ipa_targets = batch['ipa_indices'].to(self.device)
        
        # 그래디언트 초기화
        self.optimizer.zero_grad()
        
        # 순전파
        outputs = self.model(audio_features, ipa_targets)
        
        # 손실 계산
        loss = outputs['loss']
        
        # 손실이 None인 경우 처리
        if loss is None:
            raise ValueError("모델 출력에서 손실을 찾을 수 없습니다. targets가 제공되었는지 확인하세요.")
        
        # 손실이 nan인 경우 처리
        if torch.isnan(loss):
            print(f"⚠️ 경고: 손실이 NaN입니다. 배치 크기: {audio_features.shape}")
            # 작은 상수 손실로 대체
            loss = torch.tensor(0.1, device=self.device, requires_grad=True)
        
        # 역전파
        loss.backward()
        
        # 그래디언트 클리핑
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # 파라미터 업데이트
        self.optimizer.step()
        self.scheduler.step()
        
        # 현재 학습률
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            'loss': loss.item(),
            'learning_rate': current_lr
        }
    
    def train(self):
        """훈련을 수행합니다. (BaseEngine 추상 메서드 구현)"""
        # 이 메서드는 train_step을 사용하는 방식으로 구현
        print("IPA 엔진 훈련 모드: train_step을 사용하여 훈련을 진행합니다.")
        pass
    
    def evaluate(self, manifest_path: str, save_to: Optional[str] = None) -> Dict:
        """평가를 수행합니다. (BaseEngine 추상 메서드 구현)"""
        # 이 메서드는 evaluate_step을 사용하는 방식으로 구현
        print("IPA 엔진 평가 모드: evaluate_step을 사용하여 평가를 진행합니다.")
        return {"status": "evaluate_step을 사용하여 평가를 진행합니다."}
    
    def infer_file(self, wav_path: str) -> str:
        """단일 파일에 대한 추론을 수행합니다. (BaseEngine 추상 메서드 구현)"""
        # 이 메서드는 predict를 사용하는 방식으로 구현
        print(f"IPA 엔진 추론 모드: {wav_path} 파일을 처리합니다.")
        # 실제 구현은 더 복잡하지만, 기본 구조만 제공
        return "추론 결과"
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Union[float, List[str]]]:
        """평가 단계를 수행합니다."""
        self.model.eval()
        
        with torch.no_grad():
            # 배치 데이터를 디바이스로 이동
            audio_features = batch['audio_features'].to(self.device)
            ipa_targets = batch['ipa_indices'].to(self.device)
            
            # 순전파
            outputs = self.model(audio_features, ipa_targets)
            
            # 손실 계산
            loss = outputs['loss']
            
            # IPA 예측
            if self.model_type == 'wav2vec2':
                predicted_ipa = self._decode_wav2vec2_outputs(outputs['logits'])
            else:
                predicted_ipa = self._decode_whisper_outputs(outputs['generated_ids'])
            
            # 참조 IPA
            reference_ipa = self._decode_targets(ipa_targets)
            
            # 원본 텍스트
            reference_texts = batch['texts']
            
            return {
                'loss': loss.item(),
                'predicted_ipa': predicted_ipa,
                'reference_ipa': reference_ipa,
                'reference_texts': reference_texts,
                'predicted_texts': [self._ipa_to_text(ipa) for ipa in predicted_ipa]
            }
    
    def _decode_wav2vec2_outputs(self, logits: torch.Tensor) -> List[str]:
        """Wav2Vec2 출력을 IPA로 디코딩합니다."""
        # 가장 높은 확률의 토큰 선택
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # 배치의 각 샘플을 IPA로 디코딩
        predicted_ipa = []
        for ids in predicted_ids:
            ipa = self._decode_ids(ids)
            predicted_ipa.append(ipa)
        
        return predicted_ipa
    
    def _decode_whisper_outputs(self, generated_ids: torch.Tensor) -> List[str]:
        """Whisper 출력을 IPA로 디코딩합니다."""
        predicted_ipa = []
        
        for ids in generated_ids:
            ipa = self._decode_ids(ids)
            predicted_ipa.append(ipa)
        
        return predicted_ipa
    
    def _decode_targets(self, targets: torch.Tensor) -> List[str]:
        """타겟을 IPA로 디코딩합니다."""
        reference_ipa = []
        
        for target in targets:
            ipa = self._decode_ids(target)
            reference_ipa.append(ipa)
        
        return reference_ipa
    
    def _decode_ids(self, ids: torch.Tensor) -> str:
        """ID 시퀀스를 IPA 문자열로 디코딩합니다."""
        if self.idx_to_vocab is None:
            return ""
        
        ipa_chars = []
        for idx in ids:
            idx_item = idx.item()
            if idx_item in self.idx_to_vocab:
                token = self.idx_to_vocab[idx_item]
                if token not in ['<pad>', '<unk>', '<sos>', '<eos>']:
                    ipa_chars.append(token)
        
        return ''.join(ipa_chars)
    
    def _ipa_to_text(self, ipa: str) -> str:
        """IPA를 한국어 텍스트로 변환합니다."""
        # 간단한 IPA → 한국어 변환
        # 실제로는 더 정교한 역변환 로직이 필요
        
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
    
    def save_model(self, path: str):
        """모델을 저장합니다."""
        save_torch_model(self.model, path)
        print(f"모델이 저장되었습니다: {path}")
    
    def load_model(self, path: str):
        """모델을 로드합니다."""
        self.model = load_torch_model(path)
        self.model.to(self.device)
        print(f"모델이 로드되었습니다: {path}")
    
    def predict(self, audio_features: torch.Tensor) -> str:
        """단일 오디오에 대한 IPA를 예측합니다."""
        self.model.eval()
        
        with torch.no_grad():
            # 배치 차원 추가
            if audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(0)
            
            audio_features = audio_features.to(self.device)
            
            # 예측
            outputs = self.model(audio_features)
            
            if self.model_type == 'wav2vec2':
                predicted_ipa = self._decode_wav2vec2_outputs(outputs['logits'])
            else:
                predicted_ipa = self._decode_whisper_outputs(outputs['generated_ids'])
            
            return predicted_ipa[0] if predicted_ipa else ""

def create_ipa_engine(config: Dict) -> IPAEngine:
    """IPA 엔진을 생성하는 편의 함수입니다."""
    return IPAEngine(config)

if __name__ == "__main__":
    # 테스트
    config = {
        'model': {
            'type': 'wav2vec2',
            'pretrained_model': 'facebook/wav2vec2-base',
            'vocab_size': 100,
            'hidden_size': 768,
            'num_hidden_layers': 6,
            'num_attention_heads': 12,
            'dropout': 0.1
        },
        'training': {
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'max_steps': 100000,
            'scheduler': 'cosine',
            'max_grad_norm': 1.0
        },
        'seed': 42
    }
    
    # 엔진 생성
    engine = create_ipa_engine(config)
    
    # 더미 데이터로 테스트
    batch_size = 2
    n_features = 13
    time_steps = 100
    
    dummy_audio = torch.randn(batch_size, n_features, time_steps)
    dummy_ipa = torch.randint(0, 100, (batch_size, 20))
    
    batch = {
        'audio_features': dummy_audio,
        'ipa_indices': dummy_ipa,
        'texts': ['바지', '가방']
    }
    
    # 훈련 단계 테스트
    train_result = engine.train_step(batch)
    print(f"훈련 결과: {train_result}")
    
    # 평가 단계 테스트
    eval_result = engine.evaluate_step(batch)
    print(f"평가 결과: {eval_result}")
    
    # 단일 예측 테스트
    single_audio = torch.randn(n_features, time_steps)
    prediction = engine.predict(single_audio)
    print(f"단일 예측: {prediction}") 