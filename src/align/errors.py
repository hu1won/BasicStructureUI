"""
강제 정렬 관련 오류 처리를 위한 예외 클래스들
"""

class ForcedAlignmentError(Exception):
    """강제 정렬 관련 기본 예외 클래스"""
    pass

class AudioFileError(ForcedAlignmentError):
    """오디오 파일 관련 오류"""
    pass

class TranscriptionError(ForcedAlignmentError):
    """전사본 관련 오류"""
    pass

class AlignmentError(ForcedAlignmentError):
    """정렬 과정에서 발생하는 오류"""
    pass

class ModelError(ForcedAlignmentError):
    """모델 관련 오류"""
    pass

class ValidationError(ForcedAlignmentError):
    """데이터 검증 오류"""
    pass

class ConfigurationError(ForcedAlignmentError):
    """설정 관련 오류"""
    pass

def handle_alignment_error(error: Exception, context: str = "") -> str:
    """정렬 오류를 처리하고 사용자 친화적인 메시지를 반환합니다.
    
    Args:
        error (Exception): 발생한 오류
        context (str): 오류 발생 컨텍스트
    
    Returns:
        str: 사용자 친화적인 오류 메시지
    """
    if isinstance(error, AudioFileError):
        return f"오디오 파일 오류: {str(error)}"
    elif isinstance(error, TranscriptionError):
        return f"전사본 오류: {str(error)}"
    elif isinstance(error, AlignmentError):
        return f"정렬 오류: {str(error)}"
    elif isinstance(error, ModelError):
        return f"모델 오류: {str(error)}"
    elif isinstance(error, ValidationError):
        return f"검증 오류: {str(error)}"
    elif isinstance(error, ConfigurationError):
        return f"설정 오류: {str(error)}"
    else:
        return f"알 수 없는 오류: {str(error)}"

def validate_audio_file(file_path: str) -> bool:
    """오디오 파일의 유효성을 검사합니다.
    
    Args:
        file_path (str): 오디오 파일 경로
    
    Returns:
        bool: 유효성 여부
    
    Raises:
        AudioFileError: 파일이 유효하지 않은 경우
    """
    import os
    
    if not os.path.exists(file_path):
        raise AudioFileError(f"파일이 존재하지 않습니다: {file_path}")
    
    if not file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise AudioFileError(f"지원하지 않는 오디오 형식입니다: {file_path}")
    
    if os.path.getsize(file_path) == 0:
        raise AudioFileError(f"빈 파일입니다: {file_path}")
    
    return True

def validate_transcription(transcription: str) -> bool:
    """전사본의 유효성을 검사합니다.
    
    Args:
        transcription (str): 전사본 텍스트
    
    Returns:
        bool: 유효성 여부
    
    Raises:
        TranscriptionError: 전사본이 유효하지 않은 경우
    """
    if not transcription or not transcription.strip():
        raise TranscriptionError("전사본이 비어있습니다")
    
    if len(transcription.strip()) < 2:
        raise TranscriptionError("전사본이 너무 짧습니다")
    
    # 한국어 문자가 포함되어 있는지 확인
    if not any('\uAC00' <= char <= '\uD7AF' for char in transcription):
        raise TranscriptionError("한국어 문자가 포함되어 있지 않습니다")
    
    return True

def validate_alignment_config(config: dict) -> bool:
    """정렬 설정의 유효성을 검사합니다.
    
    Args:
        config (dict): 정렬 설정
    
    Returns:
        bool: 유효성 여부
    
    Raises:
        ConfigurationError: 설정이 유효하지 않은 경우
    """
    required_keys = ['model_path', 'sample_rate', 'window_size']
    
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"필수 설정 키가 누락되었습니다: {key}")
    
    if config['sample_rate'] <= 0:
        raise ConfigurationError("샘플 레이트는 양수여야 합니다")
    
    if config['window_size'] <= 0:
        raise ConfigurationError("윈도우 크기는 양수여야 합니다")
    
    return True
