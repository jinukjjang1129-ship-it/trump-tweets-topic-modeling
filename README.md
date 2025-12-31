# Trump Tweets, Topic Modeling, and Intraday Stock Market Reactions

## Abstract
본 프로젝트는 도널드 트럼프 전 미국 대통령의 트위터 발언이 미국 주식시장(S&P 500)의 단기(intraday) 수익률에 미치는 영향을 실증적으로 분석하는 것을 목적으로 한다.  
트윗 텍스트에 대해 LDA 토픽 모델링을 적용하여 주요 발언 주제를 식별하고, 그중 **무역 전쟁 및 관세 관련 토픽(Trade War / China & Tariffs)**을 핵심 분석 대상으로 설정하였다.  
해당 토픽의 강도가 높게 나타난 시점을 이벤트로 정의한 후, 분 단위 주가 수익률에 대한 회귀 분석과 이벤트 기반 숏 전략(backtesting)을 수행하였다.  
분석 결과, 특정 토픽 발언은 단기 수익률에 통계적으로 유의미한 영향을 미치며, 이를 활용한 단기 전략은 Buy & Hold 전략과 상이한 누적 성과 패턴을 보이는 것으로 나타났다.

---

## 1. 연구 목적 및 배경
정치적 발언은 금융시장에 즉각적으로 반영될 수 있는 정보 중 하나이나, 기존 연구는 주로 일 단위 데이터 또는 단순 감성 분석에 집중되어 왔다.  
본 프로젝트는 고빈도 금융 데이터와 텍스트 마이닝 기법을 결합하여, 특정 정치적 발언 주제가 시장에 미치는 **단기 반응을 분 단위 수준에서 정밀하게 분석**하는 것을 목표로 한다.

---

## 2. 분석 데이터

- **트윗 데이터**  
  도널드 트럼프 트위터 발언 (UTC 기준 타임스탬프 포함)

- **금융 데이터**  
  S&P 500 분(minute) 단위 가격 데이터

- **분석 기간**  
  2016-11-08 ~ 2019-09-22

- **분석 대상 시간**  
  정규장(RTH, 09:30–16:00 ET) 데이터만 사용

※ 원본 데이터는 용량 및 라이선스 이슈로 공개하지 않으며, 본 저장소에는 분석 재현에 필요한 최소한의 정제 데이터(processed data)만 포함한다.

---

## 3. 분석 방법론

### 3.1 텍스트 전처리
- 소문자 변환
- URL, 멘션, 해시태그 제거
- 토큰화(tokenization)
- 불용어 제거(stopwords)
- Porter Stemmer 적용
- Bigram 모델 적용

### 3.2 토픽 모델링
- LDA (Latent Dirichlet Allocation) 적용
- 토픽 수: 20개
- Coherence Score 기준 모델 적합성 평가
- 주요 토픽 중 *Trade War / China & Tariffs* 토픽을 핵심 분석 대상으로 선정

### 3.3 이벤트 정의
- 트윗을 분(minute) 단위로 집계
- 토픽 확률의 분 단위 최대값 사용
- 토픽 강도가 중앙값(median) 이상인 시점을 이벤트로 정의
- 이벤트 간 중복(overlap)을 방지하기 위해 최소 시간 간격(MIN_GAP) 적용

### 3.4 회귀 분석
- **종속변수**: 이벤트 이후 1, 10, 20, 30, 45, 60, 120분 수익률
- **독립변수**: 토픽 강도
- Robust Regression (HuberT) 적용
- 토픽 강도와 단기 수익률 간의 관계 및 통계적 유의성 검증

### 3.5 전략 백테스팅
- Trade War 토픽 이벤트 발생 시 1시간 숏(short) 포지션 가정
- S&P 500 Buy & Hold 전략과 누적 성과 비교
- 서로 다른 기준 시점에서 리베이스된 성과 곡선 생성

---

## 4. 저장소 구조

```bash
├─ README.md
├─ data
│ └─ processed
│   ├─ backtest_input.csv
│   └─ reg_input_topic0_minute.csv
│
├─ code
│ └─ main
│   ├─ backtest.py
│   └─ regression.py
│
└─ results
  └─ (회귀 결과 CSV 및 전략 성과 그래프)
```


## 5. 주요 스크립트 설명

### backtest.py
- Trade War 토픽 기반 이벤트 숏 전략과 Buy & Hold 전략의 누적 성과 비교
- 두 가지 기준 시점에서 리베이스된 성과 곡선 생성
- 결과 그래프는 `results/` 폴더에 저장

### regression.py
- minute-level 토픽 강도 데이터와 가격 수익률 데이터를 이용한 회귀 분석 수행
- 여러 horizon(1–120분)에 대해 계수 및 p-value 산출
- 회귀 결과를 CSV 파일 형태로 저장

---

## 6. 실행 방법

```bash
pip install -r requirements.txt
python code/main/backtest.py
python code/main/regression.py
```
---
 
# 7. 결론 및 의의 

본 프로젝트는 정치적 텍스트 정보를 토픽 수준에서 정량화하고, 이를 고빈도 금융 데이터와 결합함으로써 정치적 발언이 금융시장에 미치는 단기적 영향을 실증적으로 분석하였다.
또한 분석 결과를 단순 통계 검증에 그치지 않고 이벤트 기반 전략(backtesting)으로 확장하여, 연구 결과의 실무적 해석 가능성을 제시한다.
