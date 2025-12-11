"""
IT 아웃소싱 플랫폼 더미데이터 생성 모듈
실제 팀 프로젝트 데이터 특성을 반영한 고품질 더미데이터 생성
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def generate_transactions_data(n_transactions=10000, n_customers=5000, seed=42):
    """
    거래 데이터 생성 (10,000행)
    
    Args:
        n_transactions: 생성할 거래 수
        n_customers: 고객 수
        seed: 랜덤 시드
    
    Returns:
        DataFrame: 거래 데이터
    """
    np.random.seed(seed)
    
    # 거래 ID 생성
    transaction_ids = [f"TXN_{i:07d}" for i in range(1, n_transactions + 1)]
    
    # 고객 ID 생성 (거래당 고객 할당)
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    assigned_customers = np.random.choice(customer_ids, n_transactions)
    
    # 거래 날짜 생성 (과거 6개월)
    start_date = datetime.now() - timedelta(days=180)
    transaction_dates = []
    for i in range(n_transactions):
        days_offset = np.random.randint(0, 180)
        # 최근 3개월은 더 많은 거래 (시간대별 패턴)
        if days_offset < 90:
            # 주중(월~금) 60%, 주말 40%
            base_date = start_date + timedelta(days=days_offset)
            if np.random.random() < 0.6:
                # 주중: 아침(9-12) 30%, 오후(13-17) 50%, 저녁(18-21) 20%
                hour = np.random.choice([10, 14, 19], p=[0.3, 0.5, 0.2])
            else:
                # 주말: 오후/저녁 위주
                hour = np.random.choice([14, 19], p=[0.6, 0.4])
            transaction_date = base_date.replace(hour=hour, minute=np.random.randint(0, 60))
        else:
            transaction_date = start_date + timedelta(days=days_offset)
        transaction_dates.append(transaction_date)
    
    # 거래금액 생성 (100만~3,000만원, 평균 ~340만원)
    # 로그정규분포 사용하여 실제 데이터와 유사한 분포
    sales_amount = np.random.lognormal(mean=14.5, sigma=0.8, size=n_transactions)
    sales_amount = sales_amount.clip(1000000, 30000000).round(0).astype(int)
    
    # 서비스 대분류 (5가지)
    service_categories = ['웹개발', '앱개발', '시스템개발', '디자인', '기타']
    service_category = np.random.choice(service_categories, n_transactions, p=[0.35, 0.25, 0.15, 0.15, 0.1])
    
    # 수정요청 횟수 (0~10, 평균 ~3회)
    # 포아송 분포 사용하되 최대 10으로 제한
    modification_count = np.random.poisson(lam=3, size=n_transactions)
    modification_count = modification_count.clip(0, 10)
    
    # 추가결제금액 (0~500만원)
    # 대부분 0이고, 일부만 추가결제 발생
    additional_payment = np.zeros(n_transactions)
    payment_prob = 0.25  # 25% 확률로 추가결제
    payment_mask = np.random.random(n_transactions) < payment_prob
    additional_payment[payment_mask] = np.random.lognormal(mean=12, sigma=1, size=payment_mask.sum())
    additional_payment = additional_payment.clip(0, 5000000).round(0).astype(int)
    
    # 평점 (1~5, 평균 4.2 정도)
    # 베타 분포를 사용하여 실제 평점 분포와 유사하게
    service_rating = np.random.beta(a=8, b=2, size=n_transactions) * 4 + 1
    service_rating = service_rating.clip(1, 5).round(1)
    
    # 수수료율 (6.5%, 7.5%, 9.0%)
    # 고객 세그먼트에 따라 다르게 할당 (나중에 고객 데이터와 매칭)
    fee_rates = [0.065, 0.075, 0.090]
    fee_rate = np.random.choice(fee_rates, n_transactions, p=[0.3, 0.4, 0.3])
    
    # 거래 취소 확률 계산 (상관관계 반영)
    # 기본 취소율: 1.3%
    base_cancel_prob = 0.013
    
    # 수정요청이 많을수록 취소 확률 증가 (correlation: 0.6)
    mod_factor = (modification_count / 10) * 0.6 * 0.05
    
    # 추가결제가 0이면 취소 확률 증가 (correlation: -0.5)
    payment_factor = (additional_payment == 0) * 0.5 * 0.03
    
    # 평점이 낮을수록 취소 확률 증가 (correlation: -0.55)
    rating_factor = ((5 - service_rating) / 4) * 0.55 * 0.04
    
    # 거래금액이 클수록 취소 확률 감소 (고액 거래는 신중)
    amount_factor = -((sales_amount - sales_amount.mean()) / sales_amount.std()) * 0.2 * 0.01
    
    # 취소 확률 계산
    cancel_probability = base_cancel_prob + mod_factor + payment_factor + rating_factor + amount_factor
    cancel_probability = cancel_probability.clip(0, 0.15)  # 최대 15%
    
    # 노이즈 추가
    noise = np.random.normal(0, 0.005, n_transactions)
    cancel_probability = (cancel_probability + noise).clip(0, 1)
    
    # 거래 취소 여부 결정
    transaction_canceled = (np.random.random(n_transactions) < cancel_probability).astype(int)
    
    # 취소 날짜 생성 (취소된 경우에만)
    cancellation_dates = []
    for i, (canceled, trans_date) in enumerate(zip(transaction_canceled, transaction_dates)):
        if canceled:
            # 거래일로부터 1~30일 후 취소
            cancel_days = np.random.randint(1, 31)
            cancellation_date = trans_date + timedelta(days=cancel_days)
            cancellation_dates.append(cancellation_date)
        else:
            cancellation_dates.append(None)
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'customer_id': assigned_customers,
        'transaction_date': transaction_dates,
        'sales_amount': sales_amount,
        'service_category': service_category,
        'modification_count': modification_count,
        'additional_payment': additional_payment,
        'service_rating': service_rating,
        'fee_rate': fee_rate,
        'transaction_canceled': transaction_canceled,
        'cancellation_date': cancellation_dates
    })
    
    return df


def generate_customers_data(n_customers=5000, transaction_df=None, seed=42):
    """
    고객 데이터 생성 (5,000행)
    
    Args:
        n_customers: 생성할 고객 수
        transaction_df: 거래 데이터프레임 (집계용)
        seed: 랜덤 시드
    
    Returns:
        DataFrame: 고객 데이터
    """
    np.random.seed(seed)
    
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    
    # 나이 (20~60)
    age = np.random.normal(35, 10, n_customers).clip(20, 60).astype(int)
    
    # 지역
    regions = ['서울', '경기', '부산', '인천', '대구', '광주', '대전', '기타']
    region = np.random.choice(regions, n_customers, p=[0.35, 0.25, 0.12, 0.08, 0.06, 0.04, 0.04, 0.06])
    
    # 가입날짜 (과거 3년)
    registration_dates = [
        datetime.now() - timedelta(days=np.random.randint(1, 1095))
        for _ in range(n_customers)
    ]
    
    # 고객 세그먼트 (프리미엄/VIP/일반)
    # 총 구매금액에 따라 세그먼트 결정 (나중에 계산)
    # 일단 임시로 할당
    customer_segment = np.random.choice(['일반', '프리미엄', 'VIP'], n_customers, p=[0.6, 0.3, 0.1])
    
    # 거래 데이터가 있으면 집계
    if transaction_df is not None:
        # 고객별 집계
        customer_stats = transaction_df.groupby('customer_id').agg({
            'sales_amount': 'sum',
            'modification_count': 'sum',
            'additional_payment': 'sum',
            'transaction_canceled': 'sum',
            'transaction_date': ['min', 'max', 'count']
        }).reset_index()
        
        customer_stats.columns = [
            'customer_id', 'total_purchase_amount', 'total_modification_count',
            'total_additional_payment', 'total_cancellations',
            'first_transaction_date', 'last_transaction_date', 'transaction_count'
        ]
        
        # 세그먼트 재할당 (총 구매금액 기준)
        purchase_quantiles = customer_stats['total_purchase_amount'].quantile([0.7, 0.9])
        customer_stats['customer_segment'] = '일반'
        customer_stats.loc[
            customer_stats['total_purchase_amount'] >= purchase_quantiles[0.7], 'customer_segment'
        ] = '프리미엄'
        customer_stats.loc[
            customer_stats['total_purchase_amount'] >= purchase_quantiles[0.9], 'customer_segment'
        ] = 'VIP'
        
        # 모든 고객 ID에 대해 데이터 생성
        all_customers_df = pd.DataFrame({'customer_id': customer_ids})
        customers_df = all_customers_df.merge(customer_stats, on='customer_id', how='left')
        
        # 거래가 없는 고객은 기본값으로 채우기
        customers_df['total_purchase_amount'] = customers_df['total_purchase_amount'].fillna(0)
        customers_df['total_modification_count'] = customers_df['total_modification_count'].fillna(0)
        customers_df['total_additional_payment'] = customers_df['total_additional_payment'].fillna(0)
        customers_df['total_cancellations'] = customers_df['total_cancellations'].fillna(0)
        customers_df['transaction_count'] = customers_df['transaction_count'].fillna(0)
        customers_df['customer_segment'] = customers_df['customer_segment'].fillna('일반')
        
        # 나이, 지역, 가입날짜 추가
        customers_df['age'] = age
        customers_df['region'] = region
        customers_df['registration_date'] = registration_dates
        
        # 컬럼 순서 정리
        customers_df = customers_df[[
            'customer_id', 'age', 'region', 'total_purchase_amount',
            'customer_segment', 'registration_date',
            'total_modification_count', 'total_additional_payment'
        ]]
    else:
        # 거래 데이터가 없으면 기본값으로 생성
        total_purchase_amount = np.random.lognormal(mean=14, sigma=1.2, size=n_customers).clip(1000000, 50000000).round(0).astype(int)
        total_modification_count = np.random.poisson(lam=5, size=n_customers)
        total_additional_payment = np.random.lognormal(mean=11, sigma=1, size=n_customers).clip(0, 5000000).round(0).astype(int)
        
        # 세그먼트 할당
        purchase_quantiles = np.percentile(total_purchase_amount, [70, 90])
        customer_segment = np.where(
            total_purchase_amount >= purchase_quantiles[1], 'VIP',
            np.where(total_purchase_amount >= purchase_quantiles[0], '프리미엄', '일반')
        )
        
        customers_df = pd.DataFrame({
            'customer_id': customer_ids,
            'age': age,
            'region': region,
            'total_purchase_amount': total_purchase_amount,
            'customer_segment': customer_segment,
            'registration_date': registration_dates,
            'total_modification_count': total_modification_count,
            'total_additional_payment': total_additional_payment
        })
    
    return customers_df


def generate_predictions_data(transaction_df, customers_df, seed=42):
    """
    예측 타겟 데이터 생성 (churn_probability, risk_level)
    
    Args:
        transaction_df: 거래 데이터프레임
        customers_df: 고객 데이터프레임
        seed: 랜덤 시드
    
    Returns:
        DataFrame: 예측 데이터
    """
    np.random.seed(seed)
    
    # 고객별 해지 확률 계산
    # 거래 데이터에서 고객별 통계 계산
    customer_trans_stats = transaction_df.groupby('customer_id').agg({
        'modification_count': ['mean', 'sum'],
        'additional_payment': ['mean', 'sum'],
        'service_rating': 'mean',
        'transaction_canceled': 'sum',
        'transaction_date': ['min', 'max', 'count'],
        'sales_amount': 'sum'
    }).reset_index()
    
    customer_trans_stats.columns = [
        'customer_id', 'avg_modification_count', 'total_modification_count',
        'avg_additional_payment', 'total_additional_payment',
        'avg_rating', 'total_cancellations', 'first_transaction_date',
        'last_transaction_date', 'transaction_count', 'total_sales'
    ]
    
    # 거래 지속기간 계산 (일)
    customer_trans_stats['transaction_duration_days'] = (
        customer_trans_stats['last_transaction_date'] - customer_trans_stats['first_transaction_date']
    ).dt.days
    
    # 고객 데이터와 병합
    predictions_df = customers_df.merge(customer_trans_stats, on='customer_id', how='left')
    
    # 해지 확률 계산 (비즈니스 로직 반영)
    churn_probability = np.zeros(len(predictions_df))
    
    # 1. 수정요청이 많으면 → 해지 확률 증가 (80% 이상)
    high_mod_mask = predictions_df['avg_modification_count'] >= 5
    churn_probability[high_mod_mask] += 0.8
    
    # 2. 추가결제가 0이면 → 해지 확률 증가 (70% 이상)
    no_payment_mask = predictions_df['total_additional_payment'] == 0
    churn_probability[no_payment_mask] += 0.7
    
    # 3. 거래지속기간이 짧으면 → 해지 확률 증가
    short_duration_mask = predictions_df['transaction_duration_days'] < 30
    churn_probability[short_duration_mask] += 0.5
    
    # 4. 평점이 낮으면 → 해지 확률 증가
    low_rating_mask = predictions_df['avg_rating'] < 3.0
    churn_probability[low_rating_mask] += 0.6
    
    # 5. VIP/프리미엄 고객 → 해지 확률 감소 (20% 이하)
    premium_mask = predictions_df['customer_segment'].isin(['프리미엄', 'VIP'])
    churn_probability[premium_mask] *= 0.2
    
    # 6. 취소 이력이 있으면 → 해지 확률 증가
    has_cancellation_mask = predictions_df['total_cancellations'] > 0
    churn_probability[has_cancellation_mask] += 0.3
    
    # 정규화 및 노이즈 추가
    churn_probability = churn_probability.clip(0, 1)
    noise = np.random.normal(0, 0.05, len(predictions_df))
    churn_probability = (churn_probability + noise).clip(0, 1)
    
    # 리스크 레벨 할당
    risk_level = pd.cut(
        churn_probability,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['낮음', '중간', '높음']
    )
    
    predictions_df['churn_probability'] = churn_probability.round(4)
    predictions_df['risk_level'] = risk_level
    
    # 필요한 컬럼만 선택
    predictions_df = predictions_df[['customer_id', 'churn_probability', 'risk_level']]
    
    return predictions_df


def generate_dummy_data(n_transactions=10000, n_customers=5000, seed=42):
    """
    전체 더미데이터 생성
    
    Args:
        n_transactions: 거래 수
        n_customers: 고객 수
        seed: 랜덤 시드
    
    Returns:
        tuple: (transactions_df, customers_df, predictions_df)
    """
    print("=" * 60)
    print("더미데이터 생성 시작")
    print("=" * 60)
    
    # 1. 거래 데이터 생성
    print("\n[1/3] 거래 데이터 생성 중...")
    transactions_df = generate_transactions_data(n_transactions, n_customers, seed)
    print(f"✓ 거래 데이터 생성 완료: {transactions_df.shape}")
    
    # 2. 고객 데이터 생성 (거래 데이터 기반 집계)
    print("\n[2/3] 고객 데이터 생성 중...")
    customers_df = generate_customers_data(n_customers, transactions_df, seed)
    print(f"✓ 고객 데이터 생성 완료: {customers_df.shape}")
    
    # 3. 예측 데이터 생성
    print("\n[3/3] 예측 데이터 생성 중...")
    predictions_df = generate_predictions_data(transactions_df, customers_df, seed)
    print(f"✓ 예측 데이터 생성 완료: {predictions_df.shape}")
    
    print("\n" + "=" * 60)
    print("더미데이터 생성 완료!")
    print("=" * 60)
    
    return transactions_df, customers_df, predictions_df


def save_to_csv(transactions_df, customers_df, predictions_df, output_dir="data"):
    """
    데이터를 CSV 파일로 저장
    
    Args:
        transactions_df: 거래 데이터프레임
        customers_df: 고객 데이터프레임
        predictions_df: 예측 데이터프레임
        output_dir: 출력 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCSV 파일 저장 중... (경로: {output_path})")
    
    transactions_df.to_csv(output_path / "transactions.csv", index=False, encoding='utf-8-sig')
    print(f"✓ transactions.csv 저장 완료")
    
    customers_df.to_csv(output_path / "customers.csv", index=False, encoding='utf-8-sig')
    print(f"✓ customers.csv 저장 완료")
    
    predictions_df.to_csv(output_path / "predictions.csv", index=False, encoding='utf-8-sig')
    print(f"✓ predictions.csv 저장 완료")
    
    print("\n모든 CSV 파일 저장 완료!")


def validate_data(transactions_df, customers_df, predictions_df):
    """
    데이터 품질 검증
    
    Args:
        transactions_df: 거래 데이터프레임
        customers_df: 고객 데이터프레임
        predictions_df: 예측 데이터프레임
    """
    print("\n" + "=" * 60)
    print("데이터 품질 검증")
    print("=" * 60)
    
    # 1. 기본 통계
    print("\n[1] 기본 통계")
    print(f"  거래 데이터: {transactions_df.shape[0]:,}행, {transactions_df.shape[1]}열")
    print(f"  고객 데이터: {customers_df.shape[0]:,}행, {customers_df.shape[1]}열")
    print(f"  예측 데이터: {predictions_df.shape[0]:,}행, {predictions_df.shape[1]}열")
    
    # 2. 거래 데이터 검증
    print("\n[2] 거래 데이터 검증")
    print(f"  거래금액 범위: {transactions_df['sales_amount'].min():,.0f}원 ~ {transactions_df['sales_amount'].max():,.0f}원")
    print(f"  거래금액 평균: {transactions_df['sales_amount'].mean():,.0f}원")
    print(f"  수정요청 평균: {transactions_df['modification_count'].mean():.2f}회")
    print(f"  수정요청 최대: {transactions_df['modification_count'].max()}회")
    print(f"  거래취소율: {transactions_df['transaction_canceled'].mean():.4f} ({transactions_df['transaction_canceled'].mean()*100:.2f}%)")
    
    # 3. 상관관계 검증
    print("\n[3] 상관관계 검증")
    corr_mod_cancel = transactions_df['modification_count'].corr(transactions_df['transaction_canceled'])
    print(f"  수정요청 ↔ 거래취소: {corr_mod_cancel:.3f} (목표: 0.5~0.7)")
    
    corr_payment_cancel = transactions_df['additional_payment'].corr(transactions_df['transaction_canceled'])
    print(f"  추가결제 ↔ 거래취소: {corr_payment_cancel:.3f} (목표: -0.4~-0.6)")
    
    corr_rating_cancel = transactions_df['service_rating'].corr(transactions_df['transaction_canceled'])
    print(f"  평점 ↔ 거래취소: {corr_rating_cancel:.3f} (목표: -0.4~-0.6)")
    
    # 4. 결측치 검증
    print("\n[4] 결측치 검증")
    missing_trans = transactions_df.isnull().sum().sum()
    missing_cust = customers_df.isnull().sum().sum()
    missing_pred = predictions_df.isnull().sum().sum()
    print(f"  거래 데이터 결측치: {missing_trans}")
    print(f"  고객 데이터 결측치: {missing_cust}")
    print(f"  예측 데이터 결측치: {missing_pred}")
    
    # 5. 세그먼트별 통계
    print("\n[5] 세그먼트별 통계")
    segment_stats = customers_df.groupby('customer_segment').agg({
        'total_purchase_amount': ['mean', 'count'],
        'total_modification_count': 'mean'
    })
    print(segment_stats)
    
    # 6. 해지 확률 분포
    print("\n[6] 해지 확률 분포")
    print(f"  평균 해지 확률: {predictions_df['churn_probability'].mean():.4f}")
    print(f"  리스크 레벨 분포:")
    print(predictions_df['risk_level'].value_counts())
    
    print("\n" + "=" * 60)
    print("검증 완료!")
    print("=" * 60)


if __name__ == "__main__":
    # 더미데이터 생성
    transactions_df, customers_df, predictions_df = generate_dummy_data(
        n_transactions=10000,
        n_customers=5000,
        seed=42
    )
    
    # 데이터 검증
    validate_data(transactions_df, customers_df, predictions_df)
    
    # CSV 저장
    save_to_csv(transactions_df, customers_df, predictions_df)
    
    print("\n✅ 모든 작업 완료!")
    print("\n생성된 파일:")
    print("  - data/transactions.csv")
    print("  - data/customers.csv")
    print("  - data/predictions.csv")

