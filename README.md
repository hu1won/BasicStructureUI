# SpeechRecognition

화자의 발화를 IPA(국제음성기호)로 변환하고 최종적으로 텍스트로 표현하기 위한 음성 인식 및 처리 시스템입니다. 다양한 모델들을 학습하고 테스트하여 최적의 성능을 찾아보는 프로젝트입니다.

## 프로젝트 구조

```
SpeechRecognition/
├── .git/                          # Git 버전 관리
├── .venv/                         # Python 가상환경
├── configs/                       # 설정 파일들
│   ├── _base.yaml                # 기본 설정
│   ├── wav2vec2_ko.yaml         # Wav2Vec2 한국어 모델 설정
│   └── whisper_tiny_ko.yaml     # Whisper Tiny 한국어 모델 설정
├── data/                          # 데이터 디렉토리
│   ├── raw/                      # 원본 데이터 (폴더 생성 필요)
│   │   ├── wav/                  # 오디오 파일
│   │   │   ├── spk001_utt01.wav  # 화자1 발화1
│   │   │   ├── spk002_utt02.wav  # 화자1 발화2
│   │   │   └── ...               # 화자3 발화1
│   │   └── transcripts/          # 전사본 파일
│   ├── interim/                  # 중간 처리 데이터 (폴더 생성 필요)
│   └── processed/                # 최종 처리된 데이터 (폴더 생성 필요)
├── models/                        # 학습된 모델 저장소 (폴더 생성 필요)
├── runs/                          # 실험 실행 결과(폴더 생성 필요)
├── scripts/                       # 실행 스크립트
│   ├── train.py                  # 모델 학습
│   ├── evaluate.py               # 모델 평가
│   ├── infer.py                  # 모델 추론
│   ├── make_manifest.py          # 데이터 매니페스트 생성
│   └── export_onnx.py            # ONNX 모델 내보내기
├── src/                           # 소스 코드
│   ├── align/                    # 강제 정렬 관련
│   │   ├── errors.py             # 오류 처리
│   │   └── forced_align.py       # 강제 정렬 구현
│   ├── data/                     # 데이터 처리
│   │   ├── dataset.py            # 데이터셋 클래스
│   │   └── prepare_data.py       # 데이터 전처리
│   ├── engines/                  # 모델 엔진
│   │   ├── base_engine.py        # 기본 엔진 클래스
│   │   ├── conformer_engine.py   # Conformer 모델 엔진
│   │   ├── wav2vec2_engine.py   # Wav2Vec2 모델 엔진
│   │   └── whisper_engine.py     # Whisper 모델 엔진
│   ├── ipa/                      # IPA (국제음성기호) 처리
│   │   ├── g2p_ko.py            # 한국어 Grapheme-to-Phoneme
│   │   ├── ipa_map.py            # IPA 매핑
│   │   ├── postrules.py          # 후처리 규칙
│   │   └── to_ipa.py             # IPA 변환
│   ├── metrics/                  # 평가 메트릭
│   │   ├── asr_metrics.py        # ASR 성능 메트릭
│   │   ├── diar_metrics.py       # 화자 분할 메트릭
│   │   └── phoneme_metrics.py    # 음소 수준 메트릭
│   └── utils/                    # 유틸리티 함수
│       ├── io.py                 # 입출력 유틸리티
│       ├── mlflow_utils.py       # MLflow 통합 유틸리티
│       └── seed.py               # 시드 설정 유틸리티
├── .gitignore                     # Git 무시 파일 목록
├── pyproject.toml                 # 프로젝트 설정
├── requirements.txt               # Python 의존성 (GPU 학습 시)
├── requirements-for-mac.txt       # Python 의존성 (mac에서 간단한 학습 시)
└── README.md                      # 프로젝트 설명서
```

## 프로젝트 목적

이 프로젝트는 **화자의 발화를 IPA(국제음성기호)로 변환하고 최종적으로 텍스트로 표현**하는 것을 목표로 합니다. 

### 핵심 파이프라인
1. **음성 입력** → 화자의 발화 오디오 파일
2. **음성 인식 (ASR)** → 오디오를 텍스트로 변환
3. **IPA 변환** → 한국어 텍스트를 국제음성기호로 변환
4. **강제 정렬** → 음성과 텍스트의 시간적 정렬
5. **최종 출력** → IPA 기반의 정확한 음성 표현

## 주요 기능

- **음성 인식 (ASR)**: Wav2Vec2, Whisper, Conformer 모델 지원
- **IPA 변환**: 한국어 텍스트를 정확한 국제음성기호로 변환
- **강제 정렬**: 음성과 텍스트의 시간적 정렬로 정확한 발음 분석
- **다중 모델 비교**: 다양한 모델의 성능을 비교하여 최적 모델 선택
- **성능 평가**: 음소 수준, ASR 성능, 화자 분할 등 다양한 메트릭
- **설정 관리**: YAML 기반의 모델별 설정으로 실험 관리

## 설치 및 실행

### 의존성 설치

**mac 테스트 시:**
```bash
pip install -r requirements-for-mac.txt
```

**GPU 학습 시:**
```bash
pip install -r requirements.txt
```

### 모델 학습 및 테스트

**Wav2Vec2 모델:**
```bash
python scripts/train.py --config configs/wav2vec2_ko.yaml
python scripts/evaluate.py --config configs/wav2vec2_ko.yaml
python scripts/infer.py --config configs/wav2vec2_ko.yaml
```

**Whisper 모델:**
```bash
python scripts/train.py --config configs/whisper_tiny_ko.yaml
python scripts/evaluate.py --config configs/whisper_tiny_ko.yaml
python scripts/infer.py --config configs/whisper_tiny_ko.yaml
```

### IPA 변환 테스트
```bash
# 한국어 텍스트를 IPA로 변환
python -c "from src.ipa.to_ipa import text_to_ipa; print(text_to_ipa('안녕하세요'))"
```

### 모델 성능 비교
```bash
# 여러 모델의 성능을 비교하여 최적 모델 선택
python scripts/evaluate.py --config configs/wav2vec2_ko.yaml
python scripts/evaluate.py --config configs/whisper_tiny_ko.yaml
```

