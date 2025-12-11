# IT 아웃소싱 플랫폼 고객 해지예측 Streamlit 대시보드

Decision Tree 모델을 활용한 고객 리스크 스코어 실시간 표시 대시보드

## 프로젝트 구조

```
DSAdevelop/
├── app.py                    # Streamlit 메인 애플리케이션
├── requirements.txt          # 필요한 라이브러리 목록
├── README.md                 # 프로젝트 설명서
├── data/
│   ├── __init__.py
│   ├── sample_data.py        # 샘플 데이터 생성 모듈
│   └── model.pkl             # 학습된 모델 파일 (학습 후 생성)
├── models/
│   ├── __init__.py
│   ├── predictor.py          # 모델 예측 모듈
│   └── train_model.py        # 모델 학습 스크립트
└── utils/
    ├── __init__.py
    └── visualization.py      # 시각화 유틸리티 모듈
```

## 설치 방법

1. **의존성 설치**
```bash
pip install -r requirements.txt
```

2. **모델 학습** (최초 1회)
```bash
python models/train_model.py
```

이 명령어는 `data/model.pkl` 파일을 생성합니다.

## 실행 방법

```bash
streamlit run app.py
```

브라우저에서 자동으로 대시보드가 열립니다.

## 주요 기능

### 1. 대시보드
- 전체 고객 현황 및 주요 지표
- 해지 분포 및 리스크 스코어 분포 차트
- 고위험 고객 리스트
- 특성 중요도 시각화

### 2. 개별 고객 조회
- 고객별 상세 정보 조회
- 실시간 리스크 스코어 게이지 차트
- 해지 확률 및 예측 결과

### 3. 데이터 분석
- 세그먼트별 해지율 분석
- 시간별 해지율 추이
- 특성 간 상관관계 히트맵
- 원본 데이터 조회

## 기술 스택

- **Streamlit**: 웹 대시보드 프레임워크
- **Pandas**: 데이터 처리
- **Scikit-learn**: 머신러닝 모델 (Decision Tree)
- **Plotly**: 인터랙티브 차트
- **SHAP**: 모델 해석 (선택적)

## 데이터 구조

### 고객 데이터 (Customer Data)
- customer_id: 고객 ID
- age: 나이
- gender: 성별
- region: 지역
- customer_type: 고객 유형 (개인/기업)
- subscription_type: 구독 유형 (Basic/Premium/Enterprise)
- total_spent: 총 구매액
- total_orders: 총 주문 수
- avg_order_value: 평균 주문액
- last_order_days: 마지막 주문일로부터 경과일
- support_tickets: 고객센터 문의 수
- churn: 해지 여부 (타겟 변수)

### 판매자 데이터 (Seller Data)
- seller_id: 판매자 ID
- category: 카테고리
- rating: 평점
- total_projects: 총 프로젝트 수

### 거래 데이터 (Transaction Data)
- transaction_id: 거래 ID
- customer_id: 고객 ID
- seller_id: 판매자 ID
- amount: 거래 금액
- project_type: 프로젝트 유형
- status: 거래 상태

## 모델 정보

- **알고리즘**: Decision Tree Classifier
- **특성 수**: 9개
- **목표**: 고객 해지 예측 (이진 분류)

## 사용 예시

### 샘플 데이터 생성
```python
from data.sample_data import generate_all_sample_data

customer_df, seller_df, transaction_df = generate_all_sample_data(
    n_customers=1000,
    n_sellers=200,
    n_transactions=5000
)
```

### 모델 예측
```python
from models.predictor import ChurnPredictor

predictor = ChurnPredictor()
results = predictor.predict(customer_df)
```

## 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.
