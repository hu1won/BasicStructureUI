#!/usr/bin/env python3
"""
MFCC 전용 30개 단어 학습 스크립트
CNN + Transformer 모델 사용
"""

import os
import sys
import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engines.mfcc_engine import MFCCEngine
from src.data.ipa_dataset import IPADataset
from src.metrics.word_tracking import WordPerformanceTracker
from src.utils.io import ensure_dir, load_config
from src.utils.seed import set_seed


def setup_logging(output_dir: str) -> logging.Logger:
    """로깅을 설정합니다."""
    ensure_dir(output_dir)
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_mfcc_data_loaders(config: Dict) -> tuple[DataLoader, DataLoader]:
    """MFCC 데이터 로더를 생성합니다."""
    print("MFCC 데이터로더 생성 중...")
    
    # 훈련 데이터셋
    train_dataset = IPADataset(
        manifest_path=config['data']['manifest_path'],
        config=config
    )
    print(f"MFCC 훈련 데이터셋 초기화 완료: {len(train_dataset)}개 샘플")
    
    # 검증 데이터셋
    val_dataset = IPADataset(
        manifest_path=config['data']['val_manifest_path'],
        config=config
    )
    print(f"MFCC 검증 데이터셋 초기화 완료: {len(val_dataset)}개 샘플")
    
    # 어휘 설정
    vocab = train_dataset.get_vocab()
    print(f"어휘 설정 완료: {len(vocab)}개 토큰")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0,  # 디버깅을 위해 0으로 설정
        pin_memory=False  # CPU 사용 시 False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader, vocab


def train_epoch_with_mfcc_tracking(engine: MFCCEngine, train_loader: DataLoader, 
                                  config: Dict, word_tracker) -> Dict:
    """MFCC 추적이 포함된 에포크 훈련을 수행합니다."""
    total_loss = 0.0
    total_steps = 0
    
    progress_bar = tqdm(train_loader, desc="MFCC 훈련")
    
    for step, batch in enumerate(progress_bar):
        # 훈련 단계 수행
        train_result = engine.train_step(batch)
        
        # 손실 누적
        total_loss += train_result['loss']
        total_steps += 1
        
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


def evaluate_epoch_with_mfcc_tracking(engine: MFCCEngine, val_loader: DataLoader, 
                                     config: Dict, word_tracker) -> Dict:
    """MFCC 추적이 포함된 에포크 평가를 수행합니다."""
    total_loss = 0.0
    total_samples = 0
    
    progress_bar = tqdm(val_loader, desc="MFCC 평가")
    
    with torch.no_grad():
        for batch in progress_bar:
            # 평가 단계 수행
            eval_result = engine.evaluate_step(batch)
            
            # 손실 누적
            total_loss += eval_result['loss']
            
            # MFCC 성능 추적
            for i, (ref_text, pred_ipa) in enumerate(zip(eval_result['reference_texts'], eval_result['predicted_ipa'])):
                # IPA를 텍스트로 변환하여 비교
                pred_text = engine._ipa_to_text(pred_ipa)
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


def save_mfcc_performance_report(word_tracker, output_dir: str, epoch: int):
    """MFCC 기반 단어별 성능 보고서를 저장합니다."""
    ensure_dir(output_dir)
    
    # 성능 데이터 저장
    performance_path = os.path.join(output_dir, f'mfcc_performance_epoch_{epoch}.json')
    word_tracker.save_performance_data(performance_path)
    
    # 성능 그래프 저장
    graph_path = os.path.join(output_dir, f'mfcc_performance_epoch_{epoch}.png')
    word_tracker.plot_word_performance(graph_path)
    
    # 텍스트 보고서 저장
    report_path = os.path.join(output_dir, f'mfcc_performance_epoch_{epoch}.txt')
    report = word_tracker.generate_performance_report()
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"MFCC 성능 보고서 저장 완료:")
    print(f"  - 성능 데이터: {performance_path}")
    print(f"  - 성능 그래프: {graph_path}")
    print(f"  - 텍스트 보고서: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="MFCC 전용 30개 단어 학습")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/mfcc_word_30_training.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs/mfcc_word_30_training',
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--create_manifest', 
        action='store_true',
        help='MFCC 기반 매니페스트 파일 생성'
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    ensure_dir(args.output_dir)
    
    # 로깅 설정
    logger = setup_logging(args.output_dir)
    
    # 설정 파일 로드
    print(f"설정 파일 로드 중: {args.config}")
    config = load_config(args.config)
    
    # 시드 설정
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"시드가 {seed}로 설정되었습니다.")
    
    # MLflow 실행 시작
    try:
        # 기존 실행이 있는지 확인
        if mlflow.active_run():
            print("기존 MLflow 실행을 종료했습니다.")
            mlflow.end_run()
        
        mlflow.start_run()
        mlflow.log_params(config)
    except Exception as e:
        print(f"MLflow 실행 시작 실패: {e}")
    
    print("=== MFCC 전용 30개 단어 학습 시작 ===")
    print("학습 방식: 음성 → MFCC → IPA → 텍스트")
    
    try:
        # MFCC 엔진 초기화
        print("MFCC 엔진 초기화 중...")
        engine = MFCCEngine(config)
        
        # 데이터 로더 생성
        train_loader, val_loader, vocab = create_mfcc_data_loaders(config)
        
        # 어휘 설정
        engine.set_vocab(vocab)
        
        # 단어 성능 추적기 생성
        target_words = config['word_tracking']['target_words']
        word_tracker = WordPerformanceTracker(target_words)
        print(f"단어 성능 추적기 생성: {len(target_words)}개 단어")
        
        # 훈련 루프
        best_accuracy = 0.0
        patience_counter = 0
        max_epochs = config['training']['max_epochs']
        patience = config['training']['early_stopping_patience']
        
        for epoch in range(1, max_epochs + 1):
            print(f"\n=== 에포크 {epoch}/{max_epochs} ===")
            
            # 훈련
            print("MFCC 훈련 중...")
            try:
                train_metrics = train_epoch_with_mfcc_tracking(engine, train_loader, config, word_tracker)
                print(f"훈련 손실: {train_metrics['avg_loss']:.4f}")
            except Exception as e:
                print(f"MFCC 훈련 중 오류 발생: {e}")
                continue
            
            # 평가
            print("MFCC 평가 중...")
            try:
                eval_metrics = evaluate_epoch_with_mfcc_tracking(engine, val_loader, config, word_tracker)
                print(f"검증 손실: {eval_metrics['avg_loss']:.4f}")
                print(f"단어 정확도: {eval_metrics['word_accuracy']:.4f}")
                print(f"단어 WER: {eval_metrics['word_wer']:.4f}")
                print(f"단어 CER: {eval_metrics['word_cer']:.4f}")
            except Exception as e:
                print(f"MFCC 평가 중 오류 발생: {e}")
                continue
            
            # 성능 보고서 저장
            save_mfcc_performance_report(word_tracker, args.output_dir, epoch)
            
            # 조기 종료 체크
            current_accuracy = eval_metrics['word_accuracy']
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                patience_counter = 0
                
                # 최고 성능 모델 저장
                best_model_path = os.path.join(args.output_dir, 'best_mfcc_model.pth')
                torch.save(engine.model.state_dict(), best_model_path)
                print(f"최고 성능 모델 저장: {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"조기 종료: {patience} 에포크 동안 성능 향상 없음")
                    break
        
        # 최종 모델 저장
        final_model_path = os.path.join(args.output_dir, 'final_mfcc_model.pth')
        torch.save(engine.model.state_dict(), final_model_path)
        print(f"모델이 저장되었습니다: {final_model_path}")
        
        # MLflow에 모델 저장
        try:
            mlflow.pytorch.log_model(engine.model, "final_mfcc_model")
            print(f"최종 MFCC 모델 저장: {final_model_path}")
        except Exception as e:
            print(f"MLflow 모델 저장 실패: {e}")
        
        # 최종 성능 보고서 저장
        save_mfcc_performance_report(word_tracker, args.output_dir, 'final')
        
        print("\n=== MFCC 전용 30개 단어 훈련 완료 ===")
        print("학습 방식: 음성 → MFCC → IPA → 텍스트")
        print(f"최고 단어 정확도: {best_accuracy:.4f}")
        print(f"출력 디렉토리: {args.output_dir}")
        
    except Exception as e:
        print(f"MFCC 훈련 중 오류 발생: {e}")
        raise
    
    finally:
        # MLflow 실행 종료
        if mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main() 