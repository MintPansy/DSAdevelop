# Step 0: 더미데이터 생성 가이드

## 실행 순서

### 1. 더미데이터 생성
```bash
python data/generate_dummy_data.py
```

이 명령어는 다음을 수행합니다:
- 거래 데이터 10,000행 생성
- 고객 데이터 5,000행 생성
- 예측 데이터 생성 (churn_probability, risk_level)
- CSV 파일로 저장 (`data/transactions.csv`, `data/customers.csv`, `data/predictions.csv`)

### 2. 데이터 검증 및 시각화
```bash
python data/validate_and_visualize.py
```

이 명령어는 다음을 수행합니다:
- 데이터 품질 검증 리포트 생성 (`data/validation_report.txt`)
- 시각화 그래프 생성 (`data/validation_report/` 폴더)

## 생성되는 파일

### CSV 파일
- `data/transactions.csv` - 거래 데이터 (10,000행)
- `data/customers.csv` - 고객 데이터 (5,000행)
- `data/predictions.csv` - 예측 데이터 (5,000행)

### 검증 리포트
- `data/validation_report.txt` - 텍스트 검증 리포트
- `data/validation_report/*.png` - 시각화 그래프 (8개)

## 데이터 구조

### 거래 데이터 (transactions.csv)
- `transaction_id`: 거래 고유ID
- `customer_id`: 고객 고유ID
- `transaction_date`: 거래날짜
- `sales_amount`: 거래금액 (100만~3,000만원)
- `service_category`: 서비스 대분류 (웹개발, 앱개발, 시스템개발, 디자인, 기타)
- `modification_count`: 수정요청 횟수 (0~10)
- `additional_payment`: 추가결제금액 (0~500만원)
- `service_rating`: 평점 (1~5)
- `fee_rate`: 수수료율 (6.5%, 7.5%, 9.0%)
- `transaction_canceled`: 거래취소 여부 (0/1)
- `cancellation_date`: 취소날짜 (취소시에만)

### 고객 데이터 (customers.csv)
- `customer_id`: 고객 고유ID
- `age`: 나이 (20~60)
- `region`: 지역 (서울, 경기, 부산 등)
- `total_purchase_amount`: 총 구매금액
- `customer_segment`: 고객 세그먼트 (프리미엄/VIP/일반)
- `registration_date`: 가입날짜
- `total_modification_count`: 총 수정요청
- `total_additional_payment`: 총 추가결제금액

### 예측 데이터 (predictions.csv)
- `customer_id`: 고객 고유ID
- `churn_probability`: 해지 확률 (0~1)
- `risk_level`: 리스크 레벨 (높음/중간/낮음)

## 검증 체크리스트

생성 후 다음을 확인하세요:

- ✅ 행 수: transactions_df.shape[0] == 10000
- ✅ 필드명: 모든 컬럼이 정의된 대로 있나?
- ✅ 데이터 타입: numeric/categorical 맞나?
- ✅ 결측치: 없나? (NaN, None)
- ✅ 범위: 거래금액이 100만~3000만 사이나?
- ✅ 상관관계: 수정요청과 거래취소의 correlation 0.5~0.7인가?
- ✅ 해지율: 전체 거래취소율이 1~3% 사이인가?
- ✅ CSV 저장: 파일이 제대로 저장되었나?

## 비즈니스 로직 반영

다음 규칙이 데이터에 반영되어 있습니다:

1. **수정요청이 많으면** → 해지 확률 증가 (80% 이상)
2. **추가결제가 0이면** → 해지 확률 증가 (70% 이상)
3. **거래지속기간이 짧으면** → 해지 확률 증가
4. **평점이 낮으면** → 해지 확률 증가
5. **VIP/프리미엄 고객** → 해지 확률 20% 이하

## 상관관계 목표

- 수정요청 ↑ → 거래취소 확률 ↑ (correlation: 0.6)
- 추가결제 ↓ → 거래취소 확률 ↑ (correlation: -0.5)
- 평점 ↓ → 거래취소 확률 ↑ (correlation: -0.55)
- 거래지속기간 ↑ → 거래취소 확률 ↓ (correlation: -0.4)

## 문제 해결

### Python이 인식되지 않는 경우
```bash
# Python 경로 확인
where python
# 또는
python --version

# 가상환경 활성화 (필요시)
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 한글 경로 문제
프로젝트 경로에 한글이 있으면 문제가 발생할 수 있습니다. 
가능하면 영문 경로로 이동하거나, 절대 경로를 사용하세요.

### 라이브러리 설치
```bash
pip install pandas numpy matplotlib seaborn
```

