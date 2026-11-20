# 규칙 기반 감성 분석과 정보 추출

## 설치 및 실행

- 지원 환경 — Intel macOS, Linux x86_64
- 실행 기준 — 프로젝트 루트, Python 3.12

```bash
python3.12 codyssey_4_2_nlp_environment.py
source data/activate.sh
python src/korean_sentiment.py         # 영화 리뷰의 감성 점수와 계산 근거
python src/information_extractor.py    # 샘플 문장의 정규화·마스킹 결과
python -m unittest discover -s tests   # 회귀 테스트
```

- 설치 스크립트 — 가상환경·JDK·Python 패키지·사전·데이터셋 준비 및 검증. [Python 패키지 버전](requirements.txt)
- 터미널 — `source data/activate.sh`로 현재 셸 활성화
- 노트북 커널 — `.venv/bin/python` 선택

- 문장 입력 예시 — `python src/information_extractor.py "일시 2024/3/15, 참가비 50,000원."`
- 기대 결과 — 날짜 `2024-03-15`, 금액 `50000`
- 개인정보 출력 — 기본 마스킹 적용. 정규화된 값 확인 시 `--no-mask` 지정

| 평가 대상 | 확인 위치 |
|---|---|
| 감성 분석 | [05 성능 및 판정 변화](notebooks/05_korean_sentiment_scoring_evaluation.ipynb) · [06 문장별 시연](notebooks/06_sentence_sentiment_trace.ipynb) · [점수 계산과 중립 처리](감성점수.md) |
| 정보 추출 | [07 정밀도·재현율 및 실패 사례](notebooks/07_information_extraction_evaluation.ipynb) · [정규식 판독 예시](정보추출_정규표현식.md) |
| 설계 설명 | [모듈 분리·확장·회귀 검증과 접근법 비교](규칙기반_NLP_고찰.md) |

## 1. 분석 자료

| 자료·도구 | 용도 |
|---|---|
| KNU 한국어 감성사전 | 표현 14,843개와 항목별 감성 점수(−2~+2)를 제공하는 기준 사전 |
| 네이버 영화 리뷰(NSMC) | 학습 데이터 15만 건은 감성어 후보 탐색에 사용하고, 테스트 데이터 5만 건 중 1,000건은 최종 평가에 사용 |
| KoNLPy Okt | 리뷰를 형태소와 품사로 분리하고 활용형을 기본형으로 변환 |

[01 분석 자료 및 도구 탐색](notebooks/01_nlp_resources_exploration.ipynb)

## 2. 영화 도메인 감성어 선정

1. 형태소·품사별 총출현 횟수와 해당 표현을 포함한 리뷰 수 집계
2. KNU 미등록 표현 추출
3. 긍정·부정 리뷰별 출현 비율과 실제 용례를 검토해 채택 여부·점수 결정

| 선정 표현 | 긍정 리뷰 출현 건수 | 부정 리뷰 출현 건수 | 부여 점수 | 선정 근거 |
|---|---:|---:|---:|---|
| 명작 | 221 | 26 | +2 | 작품에 대한 강한 호평 |
| 여운 | 130 | 6 | +1 | 관람 후 남은 인상에 대한 호평 |
| 뻔하다 | 52 | 129 | −1 | 예측 가능한 전개에 대한 불만 |
| 쓰레기 | 30 | 391 | −2 | 작품에 대한 강한 혹평 |

- **추가 감성어 51개** — 긍정 21개, 부정 30개. 과제 기준 30개 이상 충족
- 확장 사전 규모 — KNU 원본 14,843개 포함 총 14,894개

[02 후보 탐색](notebooks/02_korean_domain_lexicon_exploration_v2.ipynb) · [03 긍정·부정 출현 비교](notebooks/03_korean_domain_sentiment_selection_v2.ipynb) · [04 검토 및 사전 생성](notebooks/04_korean_domain_lexicon_review_ui.ipynb) · [최종 선정 목록](data/lexicons/ko/derived/custom_entries_v001.csv)

## 3. 감성 점수 산출

1. 리뷰와 사전 항목의 형태소·품사 대조
2. 같은 위치에서 겹치는 후보 중 가장 긴 표현 선택
3. 표현별 부정 횟수가 홀수이면 부호 반전, 강도 조절어가 있으면 배율 적용

**예문** — “정말 명작! 전개는 조금 뻔했어요.”

| 사전 표현 | 기본 점수 | 강도 조절 | 반영 점수 |
|---|---:|---|---:|
| 명작 | +2 | 정말 × 1.5 | +3.0 |
| 뻔하다 | −1 | 조금 × 0.5 | −0.5 |

- 예문 합계 — **+3.0 − 0.5 = +2.5점 → 긍정**
- 이진 판정 — 0점 초과는 긍정, 0점 이하는 부정
- 0점 발생 조건 — 감성 표현 미검출 또는 양수·음수 상쇄

[감성 점수 계산 코드](src/korean_sentiment.py)

## 4. 성능 평가

- 평가 표본 — NSMC 테스트 리뷰 1,000건(긍정·부정 각 500건, 난수 시드 42)
- 비교 조건 — 동일 표본에 사전 2종 × 부정·강조 규칙 적용 여부
- Macro F1 — 긍정·부정 클래스별 F1의 평균
- 매칭률 — 사전 표현이 하나 이상 검출된 리뷰의 비율

| 감성사전 | 부정·강조 규칙 | 정확도 | Macro F1 | 매칭률 |
|---|---|---:|---:|---:|
| KNU 원본 | 미적용 | 67.9% | 0.678 | 84.7% |
| KNU 원본 | 적용 | 68.2% | 0.681 | 84.7% |
| 도메인 확장 | 미적용 | 69.7% | 0.696 | 86.7% |
| **도메인 확장** | **적용** | **70.2%** | **0.701** | **86.7%** |

- **확장 효과** — 동일 규칙에서 정확도 68.2% → 70.2%(+2.0%p)
- 판정 변화 — 오답 → 정답 27건, 정답 → 오답 7건
- 개선 사례 — “여운이 남는영화...”에 ‘여운’ +1점 반영. 원본 0점(부정) → 확장 +1점(긍정, 실제 정답)
- 확장 사전·규칙 적용 후 0점으로 부정 판정된 리뷰 — 221건

[05 점수 계산 및 성능 평가](notebooks/05_korean_sentiment_scoring_evaluation.ipynb) · [조건별 성능 지표](data/outputs/sentiment_evaluation/metrics_v001.csv) · [리뷰별 판정 결과](data/outputs/sentiment_evaluation/review_scores_v001.csv)
