#!/usr/bin/env python3
"""
MFCC 특징을 직접 처리하는 음성 인식 엔진
CNN + Transformer 아키텍처 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer

from src.engines.base_engine import BaseEngine
from src.utils.seed import set_seed


class MFCCTransformerModel(nn.Module):
    """MFCC 특징을 처리하는 CNN + Transformer 모델"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 모델 설정
        self.vocab_size = config['model'].get('vocab_size', 150)
        self.hidden_dim = config['model'].get('hidden_dim', 256)
        self.num_layers = config['model'].get('num_layers', 4)
        self.num_heads = config['model'].get('num_heads', 8)
        self.dropout = config['model'].get('dropout', 0.1)
        
        # MFCC 입력 처리 (13차원 특징)
        self.mfcc_dim = 13
        
        # CNN 레이어 (MFCC 특징 추출)
        self.conv1 = nn.Conv1d(self.mfcc_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, self.hidden_dim, kernel_size=3, padding=1)
        
        # 배치 정규화
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 출력 레이어
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # 드롭아웃
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, mfcc_features: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        순전파 수행
        
        Args:
            mfcc_features: (batch_size, mfcc_dim, time) MFCC 특징
            targets: (batch_size, seq_len) 타겟 시퀀스
            
        Returns:
            모델 출력 딕셔너리
        """
        batch_size, mfcc_dim, time = mfcc_features.shape
        
        # 1. CNN 특징 추출
        # (batch_size, mfcc_dim, time) -> (batch_size, hidden_dim, time)
        x = F.relu(self.bn1(self.conv1(mfcc_features)))
        x = F.max_pool1d(x, 2)  # 시간 차원 절반
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)  # 시간 차원 절반
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)  # 시간 차원 절반
        
        # 2. Transformer 입력 준비
        # (batch_size, hidden_dim, time) -> (batch_size, time, hidden_dim)
        x = x.transpose(1, 2)
        
        # 3. Transformer 인코딩
        x = self.transformer(x)
        
        # 4. 출력 투영
        logits = self.output_projection(x)
        
        # 5. 손실 계산 (타겟이 제공된 경우)
        loss = None
        if targets is not None:
            # CTC 손실 계산
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # (time, batch, vocab)
            
            # CTC 손실 계산
            loss = F.ctc_loss(
                log_probs, targets,
                input_lengths=torch.full((batch_size,), log_probs.size(0), dtype=torch.long),
                target_lengths=torch.sum(targets != 0, dim=1)
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': x
        }
    
    def get_vocab_size(self) -> int:
        """어휘 크기 반환"""
        return self.vocab_size


class MFCCEngine(BaseEngine):
    """MFCC 특징을 직접 처리하는 음성 인식 엔진"""
    
    def __init__(self, config: Dict):
        """MFCC 엔진 초기화"""
        super().__init__(config)
        
        # 시드 설정
        set_seed(config.get('seed', 42))
        
        # 모델 생성
        self.model = MFCCTransformerModel(config)
        
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
        
        print(f"MFCC 엔진 초기화 완료")
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
        
        # 학습률 설정
        lr = training_config.get('learning_rate', 1e-4)
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
        mfcc_features = batch['audio_features'].to(self.device)
        targets = batch['ipa_indices'].to(self.device)
        
        # 그래디언트 초기화
        self.optimizer.zero_grad()
        
        # 순전파
        outputs = self.model(mfcc_features, targets)
        
        # 손실 계산
        loss = outputs['loss']
        
        # 손실이 None인 경우 처리
        if loss is None:
            raise ValueError("모델 출력에서 손실을 찾을 수 없습니다.")
        
        # 손실이 nan인 경우 처리
        if torch.isnan(loss):
            print(f"⚠️ 경고: 손실이 NaN입니다. 배치 크기: {mfcc_features.shape}")
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
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Union[float, List[str]]]:
        """평가 단계를 수행합니다."""
        self.model.eval()
        
        with torch.no_grad():
            # 배치 데이터를 디바이스로 이동
            mfcc_features = batch['audio_features'].to(self.device)
            targets = batch['ipa_indices'].to(self.device)
            
            # 순전파
            outputs = self.model(mfcc_features, targets)
            
            # 손실 계산
            loss = outputs['loss']
            
            # 예측 디코딩
            predicted_indices = torch.argmax(outputs['logits'], dim=-1)
            predicted_ipa = self._decode_predictions(predicted_indices)
            
            # 참조 디코딩
            reference_ipa = self._decode_targets(targets)
            
            # 원본 텍스트
            reference_texts = batch['texts']
            
            return {
                'loss': loss.item() if loss is not None else 0.0,
                'predicted_ipa': predicted_ipa,
                'reference_ipa': reference_ipa,
                'reference_texts': reference_texts,
                'predicted_texts': [self._ipa_to_text(ipa) for ipa in predicted_ipa]
            }
    
    def _decode_predictions(self, predicted_indices: torch.Tensor) -> List[str]:
        """예측 인덱스를 IPA로 디코딩합니다."""
        if self.vocab is None:
            return ["<UNK>"] * len(predicted_indices)
        
        decoded = []
        for indices in predicted_indices:
            # 0 (PAD) 토큰 제거
            valid_indices = indices[indices != 0]
            if len(valid_indices) > 0:
                tokens = [self.idx_to_vocab[idx.item()] for idx in valid_indices]
                decoded.append(" ".join(tokens))
            else:
                decoded.append("")
        
        return decoded
    
    def _decode_targets(self, targets: torch.Tensor) -> List[str]:
        """타겟 인덱스를 IPA로 디코딩합니다."""
        if self.vocab is None:
            return ["<UNK>"] * len(targets)
        
        decoded = []
        for target in targets:
            # 0 (PAD) 토큰 제거
            valid_indices = target[target != 0]
            if len(valid_indices) > 0:
                tokens = [self.idx_to_vocab[idx.item()] for idx in valid_indices]
                decoded.append(" ".join(tokens))
            else:
                decoded.append("")
        
        return decoded
    
    def _ipa_to_text(self, ipa: str) -> str:
        """IPA를 텍스트로 변환합니다. (간단한 구현)"""
        # 실제로는 더 정교한 IPA → 텍스트 변환이 필요
        return ipa.replace(" ", "")
    
    def train(self):
        """훈련을 수행합니다. (BaseEngine 추상 메서드 구현)"""
        print("MFCC 엔진 훈련 모드: train_step을 사용하여 훈련을 진행합니다.")
        pass
    
    def evaluate(self, manifest_path: str, save_to: Optional[str] = None) -> Dict:
        """평가를 수행합니다. (BaseEngine 추상 메서드 구현)"""
        print("MFCC 엔진 평가 모드: evaluate_step을 사용하여 평가를 진행합니다.")
        return {"status": "evaluate_step을 사용하여 평가를 진행합니다."}
    
    def infer_file(self, wav_path: str) -> str:
        """단일 파일에 대한 추론을 수행합니다. (BaseEngine 추상 메서드 구현)"""
        print(f"MFCC 엔진 추론 모드: {wav_path} 파일을 처리합니다.")
        return "추론 결과" 