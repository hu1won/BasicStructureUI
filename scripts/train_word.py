#!/usr/bin/env python3
"""
단어 학습을 위한 훈련 스크립트
단어 수준의 음성 인식 모델을 훈련합니다.
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.engines.word_engine import create_word_engine
from src.data.word_dataset import create_word_dataloader
from src.utils.io import load_yaml, save_json, ensure_dir
from src.utils.seed import set_seed
from src.metrics.asr_metrics import calculate_wer, calculate_cer

def load_config(config_path: str) -> Dict:
    """설정 파일을 로드합니다."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일이 존재하지 않습니다: {config_path}")
    
    config = load_yaml(config_path)
    
    # 기본 설정과 병합
    if '_base' in config:
        base_config_path = os.path.join('configs', config['_base'])
        if os.path.exists(base_config_path):
            base_config = load_yaml(base_config_path)
            # 기본 설정을 먼저 로드하고, 현재 설정으로 덮어쓰기
            base_config.update(config)
            config = base_config
    
    return config

def setup_mlflow(config: Dict):
    """MLflow를 설정합니다."""
    mlflow_config = config.get('mlflow', {})
    
    if mlflow_config.get('enabled', True):
        mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'file:./mlruns'))
        mlflow.set_experiment(mlflow_config.get('experiment_name', 'word_training'))
        
        # 설정을 MLflow에 기록
        mlflow.log_params({
            'model_type': config['model']['type'],
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['batch_size'],
            'max_steps': config['training']['max_steps']
        })

def train_epoch(engine, train_loader: DataLoader, config: Dict) -> Dict:
    """한 에포크의 훈련을 수행합니다."""
    total_loss = 0.0
    total_steps = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc="훈련")
    
    for step, batch in enumerate(progress_bar):
        # 훈련 단계 수행
        train_result = engine.train_step(batch)
        
        # 손실 누적
        total_loss += train_result['loss']
        
        # 진행률 업데이트
        progress_bar.set_postfix({
            'loss': f"{train_result['loss']:.4f}",
            'lr': f"{train_result['learning_rate']:.6f}"
        })
        
        # MLflow에 메트릭 기록
        if mlflow.active_run():
            mlflow.log_metrics({
                'train_loss': train_result['loss'],
                'learning_rate': train_result['learning_rate']
            }, step=step)
    
    # 평균 손실 계산
    avg_loss = total_loss / total_steps
    
    return {
        'avg_loss': avg_loss,
        'total_steps': total_steps
    }

def evaluate_epoch(engine, val_loader: DataLoader, config: Dict) -> Dict:
    """한 에포크의 평가를 수행합니다."""
    total_loss = 0.0
    total_wer = 0.0
    total_cer = 0.0
    total_samples = 0
    
    all_predictions = []
    all_references = []
    
    progress_bar = tqdm(val_loader, desc="평가")
    
    with torch.no_grad():
        for batch in progress_bar:
            # 평가 단계 수행
            eval_result = engine.evaluate_step(batch)
            
            # 손실 누적
            total_loss += eval_result['loss']
            
            # 예측 텍스트와 참조 텍스트 수집
            predicted_texts = eval_result['predicted_texts']
            reference_texts = eval_result['reference_texts']
            
            all_predictions.extend(predicted_texts)
            all_references.extend(reference_texts)
            
            # WER, CER 계산
            for pred, ref in zip(predicted_texts, reference_texts):
                wer = calculate_wer(ref, pred)
                cer = calculate_cer(ref, pred)
                total_wer += wer
                total_cer += cer
                total_samples += 1
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'loss': f"{eval_result['loss']:.4f}",
                'wer': f"{total_wer/total_samples:.4f}",
                'cer': f"{total_cer/total_samples:.4f}"
            })
    
    # 평균 메트릭 계산
    avg_loss = total_loss / len(val_loader)
    avg_wer = total_wer / total_samples
    avg_cer = total_cer / total_samples
    
    return {
        'avg_loss': avg_loss,
        'avg_wer': avg_wer,
        'avg_cer': avg_cer,
        'total_samples': total_samples,
        'predictions': all_predictions,
        'references': all_references
    }

def save_checkpoint(engine, config: Dict, epoch: int, metrics: Dict, 
                   checkpoint_dir: str):
    """체크포인트를 저장합니다."""
    ensure_dir(checkpoint_dir)
    
    # 모델 저장
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    engine.save_model(checkpoint_path)
    
    # 메트릭 저장
    metrics_path = os.path.join(checkpoint_dir, f'metrics_epoch_{epoch}.json')
    save_json(metrics, metrics_path)
    
    # MLflow에 체크포인트 기록
    if mlflow.active_run():
        mlflow.pytorch.log_model(engine.model, f"checkpoint_epoch_{epoch}")
        mlflow.log_artifact(checkpoint_path)
    
    print(f"체크포인트 저장 완료: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="단어 학습 훈련")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/word_training.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs/word_training',
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='재개할 체크포인트 경로'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='랜덤 시드'
    )
    
    args = parser.parse_args()
    
    try:
        # 시드 설정
        set_seed(args.seed)
        
        # 설정 로드
        print(f"설정 파일 로드 중: {args.config}")
        config = load_config(args.config)
        
        # 출력 디렉토리 생성
        output_dir = args.output_dir
        ensure_dir(output_dir)
        
        # MLflow 설정
        setup_mlflow(config)
        
        # MLflow 실행 시작
        with mlflow.start_run():
            print("=== 단어 학습 시작 ===")
            
            # 엔진 생성
            print("엔진 초기화 중...")
            engine = create_word_engine(config)
            
            # 체크포인트에서 재개
            if args.resume:
                print(f"체크포인트에서 재개: {args.resume}")
                engine.load_model(args.resume)
            
            # 데이터로더 생성
            print("데이터로더 생성 중...")
            train_loader = create_word_dataloader(
                config['data']['train_manifest'],
                config,
                batch_size=config['training']['batch_size'],
                shuffle=True
            )
            
            val_loader = create_word_dataloader(
                config['data']['val_manifest'],
                config,
                batch_size=config['training']['batch_size'],
                shuffle=False
            )
            
            # 어휘 설정
            if hasattr(train_loader.dataset, 'get_vocab'):
                vocab = train_loader.dataset.get_vocab()
                engine.set_vocab(vocab)
                print(f"어휘 크기: {len(vocab)}")
            
            # 훈련 설정
            training_config = config['training']
            max_epochs = training_config.get('max_epochs', 100)
            eval_steps = training_config.get('eval_steps', 1000)
            save_steps = training_config.get('save_steps', 1000)
            
            # 체크포인트 디렉토리
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            ensure_dir(checkpoint_dir)
            
            # 훈련 루프
            best_wer = float('inf')
            patience_counter = 0
            early_stopping_patience = training_config.get('early_stopping_patience', 5)
            
            for epoch in range(max_epochs):
                print(f"\n=== 에포크 {epoch + 1}/{max_epochs} ===")
                
                # 훈련
                print("훈련 중...")
                train_metrics = train_epoch(engine, train_loader, config)
                
                # 평가
                print("평가 중...")
                val_metrics = evaluate_epoch(engine, val_loader, config)
                
                # 결과 출력
                print(f"훈련 손실: {train_metrics['avg_loss']:.4f}")
                print(f"검증 손실: {val_metrics['avg_loss']:.4f}")
                print(f"검증 WER: {val_metrics['avg_wer']:.4f}")
                print(f"검증 CER: {val_metrics['avg_cer']:.4f}")
                
                # MLflow에 메트릭 기록
                mlflow.log_metrics({
                    'train_loss': train_metrics['avg_loss'],
                    'val_loss': val_metrics['avg_loss'],
                    'val_wer': val_metrics['avg_wer'],
                    'val_cer': val_metrics['avg_cer']
                }, step=epoch)
                
                # 체크포인트 저장
                if (epoch + 1) % save_steps == 0:
                    save_checkpoint(engine, config, epoch + 1, val_metrics, checkpoint_dir)
                
                # 최고 성능 모델 저장
                if val_metrics['avg_wer'] < best_wer:
                    best_wer = val_metrics['avg_wer']
                    best_model_path = os.path.join(output_dir, 'best_model.pth')
                    engine.save_model(best_model_path)
                    print(f"최고 성능 모델 저장: {best_model_path}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 조기 종료 확인
                if patience_counter >= early_stopping_patience:
                    print(f"조기 종료: {early_stopping_patience} 에포크 동안 성능 향상 없음")
                    break
            
            # 최종 모델 저장
            final_model_path = os.path.join(output_dir, 'final_model.pth')
            engine.save_model(final_model_path)
            print(f"최종 모델 저장: {final_model_path}")
            
            # 훈련 완료 요약
            print("\n=== 훈련 완료 ===")
            print(f"최고 WER: {best_wer:.4f}")
            print(f"출력 디렉토리: {output_dir}")
            
            # MLflow에 최종 모델 저장
            mlflow.pytorch.log_model(engine.model, "final_model")
            
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 