# Trump Tweets & Intraday Market Reactions

**Text mining · Topic modeling · Event study · Exploratory backtesting**

도널드 트럼프의 트윗에서 주제를 추출하고, S&P 500 분봉 데이터와 연결해 발표 이후 단기 가격 반응을 탐색한 프로젝트입니다. **금융시장에 대한 인과관계나 실거래 수익성을 입증한 연구는 아닙니다.**

## 연구 질문
- 트윗 게시 여부보다 발언 **주제**가 시장 반응과 더 밀접한가?
- 무역·관세 관련 표현이 등장한 뒤의 수익률은 어떤 패턴을 보이는가?
- 관찰된 패턴을 이벤트 기반 시뮬레이션으로 나타내면 어떤 차이가 있는가?

## 분석 과정
1. 트윗 전처리: URL·멘션·불용어 처리, stemming, bigram.
2. **LDA 20개 토픽** 생성 및 coherence와 토픽 단어를 함께 검토.
3. Trade War / China & Tariffs 관련 주제를 중심으로 이벤트 선정.
4. 정규장 분봉 수익률과 연결하여 **HuberT robust regression** 적용.
5. 토픽·단어 수준의 탐색 결과를 비교하고 이벤트 기반 숏 시뮬레이션 수행.

## 핵심 결과와 해석상의 주의점
- 토픽 단위 분석에서는 일관된 통계적 유의성을 확인하지 못했습니다.
- 후속 단어 탐색에서 `tariff` 관련 낮은 p-value가 관찰됐으나, **여러 가설·시간 구간을 확인한 탐색적 결과**로서 다중검정 보정과 표본 외 검증이 필요합니다.
- 이벤트별 숏 수익률과 시장 보유 수익률을 그린 기존 그래프는 **동일한 투자 운용 조건에서 나온 두 포트폴리오의 성과 비교가 아닙니다.** 거래비용, 슬리피지, 이벤트 중복, 실제 포지션 운용을 충분히 반영하지 않았습니다.
- 따라서 기존 발표 자료의 수익률 차이는 **실거래 초과수익으로 해석하지 않습니다.**

## 저장소 구성

```text
.
├── README.md
├── src/
│   ├── regression.py          # 토픽 강도와 이후 수익률의 회귀 분석
│   └── backtest.py            # 탐색적 이벤트 숏 시뮬레이션
├── notebooks/                 # 원본 분석 노트북 (탐색 이력 보존)
├── data/
│   ├── raw/                   # 원본 트윗 및 분봉 데이터
│   └── processed/             # 일부 정제 입력 데이터
├── result/                    # 기존 발표 PDF·그래프·워드클라우드
├── docs/
│   └── methodology.md         # 가정, 제한 사항 및 재현 범위
└── requirements.txt
```

## 실행 환경

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 백테스트 시각화
```bash
python src/backtest.py
```
`data/processed/backtest_input.csv`를 읽어 `result/generated/`에 그래프를 저장합니다. 이는 **분석용 시각화**이며 투자 권유나 검증된 매매 전략이 아닙니다.

### 회귀 분석
```bash
python src/regression.py
```
현재 저장소에는 회귀 코드가 요구하는 `data/processed/reg_input_price_returns.csv`가 **없습니다.** 원본 데이터를 이용해 해당 파일을 재생성해야 실행할 수 있습니다. 재현용 전체 파이프라인은 아직 구성되지 않았습니다.

## 결과 자료
- [프로젝트 발표자료](result/ppt.pdf)
- [기존 백테스트 이미지](result/backtest_result.png)
- [토픽 워드클라우드](result/wordcloud.png)

## 향후 검증 과제
- 실험 전체에 대한 다중검정 보정 및 기간 분할 검증
- 이벤트가 겹치는 구간의 중복 계산 방지
- 거래비용·슬리피지·실제 자본배분을 반영한 일관된 백테스트
- 전처리·이벤트 생성·수익률 계산까지 실행 가능한 재현 파이프라인

기존 분석 노트북과 원본 결과 파일은 연구 기록을 보존하기 위해 변경하지 않았습니다.
