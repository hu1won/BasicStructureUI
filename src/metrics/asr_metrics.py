from jiwer import wer, cer

def compute_metrics(refs, hyps):
    return {
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
    }
