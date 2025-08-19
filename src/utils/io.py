"""
파일 입출력 및 데이터 처리를 위한 유틸리티 함수들
"""

import os
import json
import pickle
import yaml
import csv
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import numpy as np
import torch

def ensure_dir(directory: str) -> str:
    """디렉토리가 존재하지 않으면 생성합니다.
    
    Args:
        directory (str): 디렉토리 경로
    
    Returns:
        str: 생성된 디렉토리 경로
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def save_json(data: Any, file_path: str, ensure_ascii: bool = False, indent: int = 2):
    """데이터를 JSON 파일로 저장합니다.
    
    Args:
        data: 저장할 데이터
        file_path (str): 저장할 파일 경로
        ensure_ascii (bool): ASCII 인코딩 강제 여부
        indent (int): 들여쓰기 크기
    """
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)

def load_json(file_path: str) -> Any:
    """JSON 파일을 로드합니다.
    
    Args:
        file_path (str): 로드할 파일 경로
    
    Returns:
        로드된 데이터
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_yaml(data: Any, file_path: str, default_flow_style: bool = False):
    """데이터를 YAML 파일로 저장합니다.
    
    Args:
        data: 저장할 데이터
        file_path (str): 저장할 파일 경로
        default_flow_style (bool): YAML 플로우 스타일 사용 여부
    """
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=default_flow_style, 
                  allow_unicode=True, sort_keys=False)

def load_yaml(file_path: str) -> Any:
    """YAML 파일을 로드합니다.
    
    Args:
        file_path (str): 로드할 파일 경로
    
    Returns:
        로드된 데이터
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_pickle(data: Any, file_path: str):
    """데이터를 pickle 파일로 저장합니다.
    
    Args:
        data: 저장할 데이터
        file_path (str): 저장할 파일 경로
    """
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path: str) -> Any:
    """pickle 파일을 로드합니다.
    
    Args:
        file_path (str): 로드할 파일 경로
    
    Returns:
        로드된 데이터
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_torch_model(model: torch.nn.Module, file_path: str, **kwargs):
    """PyTorch 모델을 저장합니다.
    
    Args:
        model (torch.nn.Module): 저장할 모델
        file_path (str): 저장할 파일 경로
        **kwargs: torch.save의 추가 인자들
    """
    ensure_dir(os.path.dirname(file_path))
    torch.save(model.state_dict(), file_path, **kwargs)

def load_torch_model(model: torch.nn.Module, file_path: str, **kwargs) -> torch.nn.Module:
    """PyTorch 모델을 로드합니다.
    
    Args:
        model (torch.nn.Module): 로드할 모델 구조
        file_path (str): 로드할 파일 경로
        **kwargs: torch.load의 추가 인자들
    
    Returns:
        torch.nn.Module: 가중치가 로드된 모델
    """
    model.load_state_dict(torch.load(file_path, **kwargs))
    return model

def save_numpy_array(array: np.ndarray, file_path: str):
    """NumPy 배열을 저장합니다.
    
    Args:
        array (np.ndarray): 저장할 배열
        file_path (str): 저장할 파일 경로
    """
    ensure_dir(os.path.dirname(file_path))
    np.save(file_path, array)

def load_numpy_array(file_path: str) -> np.ndarray:
    """NumPy 배열을 로드합니다.
    
    Args:
        file_path (str): 로드할 파일 경로
    
    Returns:
        np.ndarray: 로드된 배열
    """
    return np.load(file_path)

def save_csv(data: List[Dict], file_path: str, fieldnames: Optional[List[str]] = None):
    """데이터를 CSV 파일로 저장합니다.
    
    Args:
        data (List[Dict]): 저장할 데이터 리스트
        file_path (str): 저장할 파일 경로
        fieldnames (Optional[List[str]]): CSV 헤더 필드명
    """
    ensure_dir(os.path.dirname(file_path))
    
    if not fieldnames and data:
        fieldnames = list(data[0].keys())
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def load_csv(file_path: str) -> List[Dict]:
    """CSV 파일을 로드합니다.
    
    Args:
        file_path (str): 로드할 파일 경로
    
    Returns:
        List[Dict]: 로드된 데이터 리스트
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def get_file_extension(file_path: str) -> str:
    """파일의 확장자를 반환합니다.
    
    Args:
        file_path (str): 파일 경로
    
    Returns:
        str: 파일 확장자 (점 포함)
    """
    return Path(file_path).suffix

def get_file_name_without_extension(file_path: str) -> str:
    """확장자를 제외한 파일명을 반환합니다.
    
    Args:
        file_path (str): 파일 경로
    
    Returns:
        str: 확장자 제외 파일명
    """
    return Path(file_path).stem

def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> List[str]:
    """디렉토리 내의 파일들을 리스트로 반환합니다.
    
    Args:
        directory (str): 디렉토리 경로
        pattern (str): 파일 패턴 (glob 형식)
        recursive (bool): 재귀적 검색 여부
    
    Returns:
        List[str]: 파일 경로 리스트
    """
    path = Path(directory)
    if recursive:
        files = path.rglob(pattern)
    else:
        files = path.glob(pattern)
    
    return [str(f) for f in files if f.is_file()]

def get_file_size(file_path: str) -> int:
    """파일 크기를 바이트 단위로 반환합니다.
    
    Args:
        file_path (str): 파일 경로
    
    Returns:
        int: 파일 크기 (바이트)
    """
    return os.path.getsize(file_path)

def format_file_size(size_bytes: int) -> str:
    """바이트 크기를 사람이 읽기 쉬운 형태로 변환합니다.
    
    Args:
        size_bytes (int): 바이트 크기
    
    Returns:
        str: 포맷된 파일 크기
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def backup_file(file_path: str, backup_suffix: str = ".backup") -> str:
    """파일을 백업합니다.
    
    Args:
        file_path (str): 백업할 파일 경로
        backup_suffix (str): 백업 파일 접미사
    
    Returns:
        str: 백업 파일 경로
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"백업할 파일이 존재하지 않습니다: {file_path}")
    
    backup_path = file_path + backup_suffix
    backup_counter = 1
    
    while os.path.exists(backup_path):
        backup_path = f"{file_path}.{backup_suffix}{backup_counter}"
        backup_counter += 1
    
    import shutil
    shutil.copy2(file_path, backup_path)
    return backup_path

def safe_save(data: Any, file_path: str, save_func, backup: bool = True, **kwargs):
    """안전하게 파일을 저장합니다 (백업 후 저장).
    
    Args:
        data: 저장할 데이터
        file_path (str): 저장할 파일 경로
        save_func: 저장 함수
        backup (bool): 백업 생성 여부
        **kwargs: save_func의 추가 인자들
    """
    if backup and os.path.exists(file_path):
        backup_file(file_path)
    
    save_func(data, file_path, **kwargs)

if __name__ == "__main__":
    # 테스트
    print("=== IO 유틸리티 테스트 ===")
    
    # 디렉토리 생성
    test_dir = "test_output"
    ensure_dir(test_dir)
    print(f"디렉토리 생성: {test_dir}")
    
    # JSON 저장/로드 테스트
    test_data = {"name": "테스트", "value": 42, "list": [1, 2, 3]}
    json_path = os.path.join(test_dir, "test.json")
    save_json(test_data, json_path)
    loaded_data = load_json(json_path)
    print(f"JSON 테스트: {loaded_data == test_data}")
    
    # YAML 저장/로드 테스트
    yaml_path = os.path.join(test_dir, "test.yaml")
    save_yaml(test_data, yaml_path)
    loaded_yaml = load_yaml(yaml_path)
    print(f"YAML 테스트: {loaded_yaml == test_data}")
    
    # 파일 정보 테스트
    print(f"파일 크기: {format_file_size(get_file_size(json_path))}")
    print(f"파일 확장자: {get_file_extension(json_path)}")
    print(f"파일명 (확장자 제외): {get_file_name_without_extension(json_path)}")
    
    # 정리
    import shutil
    shutil.rmtree(test_dir)
    print("테스트 완료")
