#!/usr/bin/env python3
"""
간단한 CNN 기반 음성 인식 학습 스크립트
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.data.ipa_dataset import IPADataset
from src.engines.simple_cnn_engine import SimpleCNNEngine
from src.utils.io import ensure_dir, load_config
from src.utils.seed import set_seed
from src.metrics.word_tracking import WordPerformanceTracker


def setup_logging(output_dir: str):
    """로깅 설정"""
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_data_loaders(config: dict):
    """데이터 로더 생성"""
    # 훈련 데이터셋
    train_dataset = IPADataset(
        manifest_path=config['data']['train_manifest'],
        config=config,
        is_training=True
    )
    
    # 검증 데이터셋
    val_dataset = IPADataset(
        manifest_path=config['data']['val_manifest'],
        config=config,
        is_training=False
    )
    
    # 데이터 로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader, train_dataset.vocab


def train_epoch(engine: SimpleCNNEngine, train_loader: DataLoader, epoch: int):
    """에포크 훈련"""
    engine.model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # 훈련 스텝
            results = engine.train_step(batch)
            loss = results['loss']
            
            total_loss += loss
            num_batches += 1
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
        except Exception as e:
            logging.error(f"배치 {batch_idx} 훈련 중 오류: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate_epoch(engine: SimpleCNNEngine, val_loader: DataLoader):
    """에포크 평가"""
    engine.model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                # 평가 스텝
                results = engine.evaluate_step(batch)
                loss = results['loss']
                accuracy = results['accuracy']
                
                total_loss += loss
                total_accuracy += accuracy
                num_batches += 1
                
            except Exception as e:
                logging.error(f"평가 중 오류: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_accuracy


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="간단한 CNN 음성 인식 학습")
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    parser.add_argument('--output_dir', type=str, required=True, help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    ensure_dir(args.output_dir)
    
    # 로깅 설정
    setup_logging(args.output_dir)
    
    # 설정 로드
    config = load_config(args.config)
    logging.info(f"설정 파일 로드 완료: {args.config}")
    
    # 시드 설정
    set_seed(config.get('seed', 42))
    logging.info("시드 설정 완료")
    
    # 데이터 로더 생성
    train_loader, val_loader, vocab = create_data_loaders(config)
    logging.info(f"데이터 로더 생성 완료: 훈련 {len(train_loader)} 배치, 검증 {len(val_loader)} 배치")
    
    # 엔진 초기화
    engine = SimpleCNNEngine(config)
    engine.set_vocab(vocab)
    
    # 훈련 설정
    max_epochs = config['training']['max_epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)
    
    # 훈련 루프
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    logging.info("=== 간단한 CNN 음성 인식 학습 시작 ===")
    
    for epoch in range(max_epochs):
        # 훈련
        train_loss = train_epoch(engine, train_loader, epoch)
        train_losses.append(train_loss)
        
        # 검증
        val_loss, val_accuracy = evaluate_epoch(engine, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 로깅
        logging.info(f"Epoch {epoch+1}/{max_epochs}")
        logging.info(f"  훈련 손실: {train_loss:.4f}")
        logging.info(f"  검증 손실: {val_loss:.4f}")
        logging.info(f"  검증 정확도: {val_accuracy:.4f}")
        
        # 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 최고 모델 저장
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            engine.save_model(best_model_path)
            logging.info(f"최고 모델 저장: {best_model_path}")
        else:
            patience_counter += 1
        
        # 조기 종료 체크
        if patience_counter >= early_stopping_patience:
            logging.info(f"조기 종료: {early_stopping_patience} 에포크 동안 성능 향상 없음")
            break
    
    # 최종 모델 저장
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    engine.save_model(final_model_path)
    logging.info(f"최종 모델 저장: {final_model_path}")
    
    # 최종 성능 요약
    logging.info("=== 학습 완료 ===")
    logging.info(f"최고 검증 손실: {best_val_loss:.4f}")
    logging.info(f"최종 검증 정확도: {val_accuracies[-1]:.4f}")
    logging.info(f"출력 디렉토리: {args.output_dir}")


if __name__ == "__main__":
    main() 