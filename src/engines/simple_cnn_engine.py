"""
간단한 CNN 기반 음성 인식 엔진
MFCC 특징을 직접 처리하는 간단하고 안정적인 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np

from ..utils.seed import set_seed


class SimpleCNNModel(nn.Module):
    """간단한 CNN 기반 음성 인식 모델"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 설정
        self.n_mfcc = config['data']['audio']['mfcc']['n_mfcc']
        self.vocab_size = config['model']['vocab_size']
        
        # CNN 레이어들
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        
        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 드롭아웃
        self.dropout = nn.Dropout(0.5)
        
        # 전역 평균 풀링 후 분류
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.vocab_size)
        )
        
    def forward(self, mfcc_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """순전파"""
        batch_size = mfcc_features.shape[0]
        
        # MFCC를 2D로 변환 (batch, 1, n_mfcc, time)
        if mfcc_features.dim() == 3:
            x = mfcc_features.unsqueeze(1)  # (batch, 1, n_mfcc, time)
        else:
            x = mfcc_features
        
        # CNN 처리
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # 전역 평균 풀링
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(batch_size, -1)
        
        # 분류
        logits = self.classifier(x)
        
        return {
            'logits': logits,
            'loss': None  # 손실은 엔진에서 계산
        }


class SimpleCNNEngine:
    """간단한 CNN 기반 음성 인식 엔진"""
    
    def __init__(self, config: Dict):
        """엔진 초기화"""
        # 시드 설정
        set_seed(config.get('seed', 42))
        
        # 모델 생성
        self.model = SimpleCNNModel(config)
        
        # 어휘 설정
        self.vocab = None
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        
        # 옵티마이저 설정
        self._setup_optimizer(config)
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"간단한 CNN 엔진 초기화 완료")
        print(f"디바이스: {self.device}")
        print(f"어휘 크기: {config['model']['vocab_size']}")
    
    def _setup_optimizer(self, config: Dict):
        """옵티마이저 설정"""
        training_config = config['training']
        
        # 학습률
        lr = training_config.get('learning_rate', 1e-3)
        if isinstance(lr, str):
            lr = float(lr)
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        # 스케줄러
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=50,
            gamma=0.9
        )
    
    def set_vocab(self, vocab: List[str]):
        """어휘 설정"""
        self.vocab = vocab
        self.vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_vocab = {idx: token for idx, token in enumerate(vocab)}
        print(f"어휘 설정 완료: {len(vocab)}개 토큰")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """훈련 단계"""
        self.model.train()
        
        # 배치 데이터를 디바이스로 이동
        mfcc_features = batch['audio_features'].to(self.device)
        targets = batch['ipa_indices'].to(self.device)
        
        # 그래디언트 초기화
        self.optimizer.zero_grad()
        
        # 순전파
        outputs = self.model(mfcc_features)
        logits = outputs['logits']
        
        # 손실 계산 (CrossEntropy)
        # targets를 1D로 변환 (batch_size,)
        if targets.dim() > 1:
            targets = targets[:, 0]  # 첫 번째 토큰만 사용
        
        loss = F.cross_entropy(logits, targets)
        
        # 역전파
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 옵티마이저 스텝
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'logits': logits.detach()
        }
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """평가 단계"""
        self.model.eval()
        
        with torch.no_grad():
            # 배치 데이터를 디바이스로 이동
            mfcc_features = batch['audio_features'].to(self.device)
            targets = batch['ipa_indices'].to(self.device)
            
            # 순전파
            outputs = self.model(mfcc_features)
            logits = outputs['logits']
            
            # 손실 계산
            if targets.dim() > 1:
                targets = targets[:, 0]
            
            loss = F.cross_entropy(logits, targets)
            
            # 예측
            predictions = torch.argmax(logits, dim=-1)
            
            # 정확도 계산
            accuracy = (predictions == targets).float().mean().item()
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy,
                'predictions': predictions.cpu(),
                'targets': targets.cpu()
            }
    
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab': self.vocab,
            'config': self.model.n_mfcc
        }, path)
        print(f"모델이 저장되었습니다: {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.vocab = checkpoint['vocab']
        print(f"모델이 로드되었습니다: {path}")
    
    def predict(self, mfcc_features: torch.Tensor) -> List[str]:
        """예측"""
        self.model.eval()
        
        with torch.no_grad():
            mfcc_features = mfcc_features.to(self.device)
            outputs = self.model(mfcc_features)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            
            # 인덱스를 텍스트로 변환
            results = []
            for pred in predictions:
                if pred.item() < len(self.vocab):
                    results.append(self.vocab[pred.item()])
                else:
                    results.append('<unk>')
            
            return results 