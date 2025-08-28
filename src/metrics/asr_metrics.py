from jiwer import wer, cer

def compute_metrics(refs, hyps):
    return {
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
    }

def calculate_wer(reference: str, prediction: str) -> float:
    """Word Error Rate를 계산합니다."""
    return wer([reference], [prediction])

def calculate_cer(reference: str, prediction: str) -> float:
    """Character Error Rate를 계산합니다."""
    return cer([reference], [prediction])
