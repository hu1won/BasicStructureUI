"""
재현 가능한 실험을 위한 시드 설정 유틸리티
"""

import os
import random
import numpy as np
import torch
from typing import Optional, Union

def set_seed(seed: int = 42, deterministic: bool = True):
    """모든 시드를 설정하여 재현 가능한 실험 환경을 만듭니다.
    
    Args:
        seed (int): 시드 값
        deterministic (bool): 결정론적 연산 사용 여부
    """
    # Python random 시드 설정
    random.seed(seed)
    
    # NumPy 시드 설정
    np.random.seed(seed)
    
    # PyTorch 시드 설정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    
    # PyTorch 결정론적 연산 설정
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 환경 변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"시드가 {seed}로 설정되었습니다.")
    if deterministic:
        print("결정론적 연산이 활성화되었습니다.")

def get_random_seed() -> int:
    """현재 설정된 시드를 반환합니다.
    
    Returns:
        int: 현재 시드 값
    """
    return random.randint(1, 1000000)

def set_torch_seed(seed: int):
    """PyTorch 관련 시드만 설정합니다.
    
    Args:
        seed (int): 시드 값
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def set_numpy_seed(seed: int):
    """NumPy 시드만 설정합니다.
    
    Args:
        seed (int): 시드 값
    """
    np.random.seed(seed)

def set_python_seed(seed: int):
    """Python random 시드만 설정합니다.
    
    Args:
        seed (int): 시드 값
    """
    random.seed(seed)

def seed_worker(worker_id: int):
    """DataLoader 워커의 시드를 설정합니다.
    
    Args:
        worker_id (int): 워커 ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader_kwargs(seed: int = 42) -> dict:
    """DataLoader에 사용할 시드 관련 키워드 인자를 반환합니다.
    
    Args:
        seed (int): 시드 값
    
    Returns:
        dict: DataLoader 키워드 인자
    """
    return {
        'worker_init_fn': seed_worker,
        'generator': torch.Generator().manual_seed(seed)
    }

def set_cuda_deterministic():
    """CUDA 결정론적 연산을 설정합니다."""
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("CUDA 결정론적 연산이 활성화되었습니다.")

def reset_seeds():
    """모든 시드를 초기화합니다."""
    random.seed()
    np.random.seed()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    print("모든 시드가 초기화되었습니다.")

class SeedContext:
    """시드 설정을 컨텍스트 매니저로 관리하는 클래스"""
    
    def __init__(self, seed: int = 42, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.original_states = {}
    
    def __enter__(self):
        # 현재 상태 저장
        self.original_states['python'] = random.getstate()
        self.original_states['numpy'] = np.random.get_state()
        self.original_states['torch'] = torch.get_rng_state()
        if torch.cuda.is_available():
            self.original_states['cuda'] = torch.cuda.get_rng_state()
        
        # 새로운 시드 설정
        set_seed(self.seed, self.deterministic)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 원래 상태로 복원
        random.setstate(self.original_states['python'])
        np.random.set_state(self.original_states['numpy'])
        torch.set_rng_state(self.original_states['torch'])
        if 'cuda' in self.original_states:
            torch.cuda.set_rng_state(self.original_states['cuda'])

def with_seed(seed: int = 42, deterministic: bool = True):
    """데코레이터로 시드를 설정하는 함수"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with SeedContext(seed, deterministic):
                return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # 테스트
    print("=== 시드 설정 테스트 ===")
    
    # 기본 시드 설정
    set_seed(42)
    
    # 랜덤 값 생성
    python_rand = random.random()
    numpy_rand = np.random.random()
    torch_rand = torch.rand(1).item()
    
    print(f"Python random: {python_rand}")
    print(f"NumPy random: {numpy_rand}")
    print(f"PyTorch random: {torch_rand}")
    
    # 시드 재설정 후 동일한 값 생성
    set_seed(42)
    python_rand2 = random.random()
    numpy_rand2 = np.random.random()
    torch_rand2 = torch.rand(1).item()
    
    print(f"\n재설정 후:")
    print(f"Python random: {python_rand2}")
    print(f"NumPy random: {numpy_rand2}")
    print(f"PyTorch random: {torch_rand2}")
    
    print(f"\n값 일치 여부:")
    print(f"Python: {python_rand == python_rand2}")
    print(f"NumPy: {numpy_rand == numpy_rand2}")
    print(f"PyTorch: {torch_rand == torch_rand2}")
    
    # 컨텍스트 매니저 테스트
    print(f"\n=== 컨텍스트 매니저 테스트 ===")
    with SeedContext(123):
        print(f"컨텍스트 내부: {random.random()}")
    
    print(f"컨텍스트 외부: {random.random()}")
    
    # 데코레이터 테스트
    @with_seed(456)
    def test_function():
        return random.random()
    
    print(f"\n데코레이터 테스트: {test_function()}")
