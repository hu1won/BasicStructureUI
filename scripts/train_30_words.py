#!/usr/bin/env python3
"""
30개 단어 학습을 위한 실행 스크립트
정해진 30개 한국어 단어를 중심으로 학습합니다.
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
from src.metrics.word_tracking import create_word_tracker

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
        mlflow.set_experiment(mlflow_config.get('experiment_name', 'word_30_training'))
        
        # 설정을 MLflow에 기록
        mlflow.log_params({
            'model_type': config['model']['type'],
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['batch_size'],
            'max_epochs': config['training']['max_epochs'],
            'target_words_count': 30
        })

def create_word_manifest():
    """30개 단어에 대한 매니페스트를 생성합니다."""
    target_words = [
        "바지", "가방", "접시", "장갑", "뽀뽀", "포크", "아프다", "단추", "침대", "숟가락",
        "꽃", "딸기", "목도리", "토끼", "코", "짹짹", "사탕", "우산", "싸우다", "눈사람",
        "휴지", "비행기", "먹다", "라면", "나무", "그네", "양말", "머리", "나비", "웃다"
    ]
    
    # 매니페스트 데이터 생성
    manifest_data = []
    
    for i, word in enumerate(target_words, 1):
        # 오디오 파일 경로
        audio_file = f"data/raw/wav/word_{i:02d}_{word}.wav"
        
        # 전사본 파일 경로
        transcript_file = f"data/raw/transcripts/word_{i:02d}_{word}.txt"
        
        # 매니페스트 항목 생성
        manifest_item = {
            'audio_file': audio_file,
            'transcript_file': transcript_file,
            'text': word,
            'word_id': i,
            'word': word,
            'duration': None,  # 나중에 계산
            'category': 'basic_vocabulary'
        }
        
        manifest_data.append(manifest_item)
    
    # 훈련/검증 분할 (8:2)
    train_data = manifest_data[:24]  # 24개 단어
    val_data = manifest_data[24:]    # 6개 단어
    
    # 전체 매니페스트
    full_manifest = {
        'train': train_data,
        'val': val_data,
        'total_samples': len(manifest_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'split_ratio': 0.8,
        'target_words': target_words
    }
    
    # 매니페스트 저장
    output_dir = "data"
    ensure_dir(output_dir)
    
    # 전체 매니페스트
    full_manifest_path = os.path.join(output_dir, "manifest_word_30.json")
    save_json(full_manifest, full_manifest_path)
    
    # 개별 매니페스트
    train_manifest_path = os.path.join(output_dir, "manifest_word_30_train.json")
    val_manifest_path = os.path.join(output_dir, "manifest_word_30_val.json")
    
    save_json(train_data, train_manifest_path)
    save_json(val_data, val_manifest_path)
    
    print(f"30개 단어 매니페스트 생성 완료:")
    print(f"  - 전체 매니페스트: {full_manifest_path}")
    print(f"  - 훈련 매니페스트: {train_manifest_path}")
    print(f"  - 검증 매니페스트: {val_manifest_path}")
    print(f"  - 훈련: {len(train_data)}개, 검증: {len(val_data)}개")
    
    return full_manifest_path, train_manifest_path, val_manifest_path

def train_epoch_with_tracking(engine, train_loader: DataLoader, config: Dict, 
                             word_tracker) -> Dict:
    """단어 추적이 포함된 에포크 훈련을 수행합니다."""
    total_loss = 0.0
    total_steps = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc="훈련")
    
    for step, batch in enumerate(progress_bar):
        # 훈련 단계 수행
        train_result = engine.train_step(batch)
        
        # 손실 누적
        total_loss += train_result['loss']
        
        # 단어 성능 추적
        for i, (ref_text, pred_text) in enumerate(zip(batch['texts'], batch.get('predicted_texts', [''] * len(batch['texts'])))):
            if pred_text:  # 예측이 있는 경우
                word_tracker.update(ref_text, pred_text, f"train_step_{step}_sample_{i}")
        
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

def evaluate_epoch_with_tracking(engine, val_loader: DataLoader, config: Dict, 
                               word_tracker) -> Dict:
    """단어 추적이 포함된 에포크 평가를 수행합니다."""
    total_loss = 0.0
    total_samples = 0
    
    progress_bar = tqdm(val_loader, desc="평가")
    
    with torch.no_grad():
        for batch in progress_bar:
            # 평가 단계 수행
            eval_result = engine.evaluate_step(batch)
            
            # 손실 누적
            total_loss += eval_result['loss']
            
            # 단어 성능 추적
            for i, (ref_text, pred_text) in enumerate(zip(eval_result['reference_texts'], eval_result['predicted_texts'])):
                word_tracker.update(ref_text, pred_text, f"val_step_{total_samples + i}")
                total_samples += 1
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'loss': f"{eval_result['loss']:.4f}",
                'samples': total_samples
            })
    
    # 평균 손실 계산
    avg_loss = total_loss / len(val_loader)
    
    # 단어별 성능 통계
    word_performance = word_tracker.get_overall_performance()
    
    return {
        'avg_loss': avg_loss,
        'total_samples': total_samples,
        'word_accuracy': word_performance['overall_accuracy'],
        'word_wer': word_performance['overall_wer'],
        'word_cer': word_performance['overall_cer']
    }

def save_word_performance_report(word_tracker, output_dir: str, epoch: int):
    """단어별 성능 보고서를 저장합니다."""
    ensure_dir(output_dir)
    
    # 성능 데이터 저장
    performance_path = os.path.join(output_dir, f'word_performance_epoch_{epoch}.json')
    word_tracker.save_performance_data(performance_path)
    
    # 성능 그래프 저장
    graph_path = os.path.join(output_dir, f'word_performance_epoch_{epoch}.png')
    word_tracker.plot_word_performance(graph_path)
    
    # 텍스트 보고서 저장
    report_path = os.path.join(output_dir, f'word_performance_epoch_{epoch}.txt')
    report = word_tracker.generate_performance_report()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"단어 성능 보고서 저장 완료:")
    print(f"  - 성능 데이터: {performance_path}")
    print(f"  - 성능 그래프: {graph_path}")
    print(f"  - 텍스트 보고서: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="30개 단어 학습")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/word_30_training.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs/word_30_training',
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--create_manifest', 
        action='store_true',
        help='매니페스트 파일 생성'
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
        
        # 매니페스트 생성 (필요시)
        if args.create_manifest:
            print("30개 단어 매니페스트 생성 중...")
            create_word_manifest()
        
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
            print("=== 30개 단어 학습 시작 ===")
            
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
            
            # 단어 성능 추적기 생성
            target_words = config.get('word_tracking', {}).get('target_words', [])
            if not target_words:
                # 기본 30개 단어
                target_words = [
                    "바지", "가방", "접시", "장갑", "뽀뽀", "포크", "아프다", "단추", "침대", "숟가락",
                    "꽃", "딸기", "목도리", "토끼", "코", "짹짹", "사탕", "우산", "싸우다", "눈사람",
                    "휴지", "비행기", "먹다", "라면", "나무", "그네", "양말", "머리", "나비", "웃다"
                ]
            
            word_tracker = create_word_tracker(target_words)
            print(f"단어 성능 추적기 생성: {len(target_words)}개 단어")
            
            # 훈련 설정
            training_config = config['training']
            max_epochs = training_config.get('max_epochs', 200)
            eval_steps = training_config.get('eval_steps', 500)
            save_steps = training_config.get('save_steps', 500)
            
            # 체크포인트 디렉토리
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            ensure_dir(checkpoint_dir)
            
            # 훈련 루프
            best_word_accuracy = 0.0
            patience_counter = 0
            early_stopping_patience = training_config.get('early_stopping_patience', 10)
            
            for epoch in range(max_epochs):
                print(f"\n=== 에포크 {epoch + 1}/{max_epochs} ===")
                
                # 훈련
                print("훈련 중...")
                train_metrics = train_epoch_with_tracking(engine, train_loader, config, word_tracker)
                
                # 평가
                print("평가 중...")
                val_metrics = evaluate_epoch_with_tracking(engine, val_loader, config, word_tracker)
                
                # 결과 출력
                print(f"훈련 손실: {train_metrics['avg_loss']:.4f}")
                print(f"검증 손실: {val_metrics['avg_loss']:.4f}")
                print(f"단어 정확도: {val_metrics['word_accuracy']:.4f}")
                print(f"단어 WER: {val_metrics['word_wer']:.4f}")
                print(f"단어 CER: {val_metrics['word_cer']:.4f}")
                
                # MLflow에 메트릭 기록
                mlflow.log_metrics({
                    'train_loss': train_metrics['avg_loss'],
                    'val_loss': val_metrics['avg_loss'],
                    'word_accuracy': val_metrics['word_accuracy'],
                    'word_wer': val_metrics['word_wer'],
                    'word_cer': val_metrics['word_cer']
                }, step=epoch)
                
                # 체크포인트 저장
                if (epoch + 1) % save_steps == 0:
                    save_checkpoint(engine, config, epoch + 1, val_metrics, checkpoint_dir)
                    save_word_performance_report(word_tracker, output_dir, epoch + 1)
                
                # 최고 성능 모델 저장
                if val_metrics['word_accuracy'] > best_word_accuracy:
                    best_word_accuracy = val_metrics['word_accuracy']
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
            
            # 최종 성능 보고서
            save_word_performance_report(word_tracker, output_dir, 'final')
            
            # 훈련 완료 요약
            print("\n=== 30개 단어 훈련 완료 ===")
            print(f"최고 단어 정확도: {best_word_accuracy:.4f}")
            print(f"출력 디렉토리: {output_dir}")
            
            # MLflow에 최종 모델 저장
            mlflow.pytorch.log_model(engine.model, "final_model")
            
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

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

if __name__ == "__main__":
    exit(main()) 