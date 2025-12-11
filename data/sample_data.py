"""
샘플 데이터 생성 모듈
IT 아웃소싱 플랫폼의 고객, 판매자, 거래 데이터를 생성합니다.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_customer_data(n_customers=1000, seed=42):
    """
    고객 데이터 생성
    
    Args:
        n_customers: 생성할 고객 수
        seed: 랜덤 시드
    
    Returns:
        DataFrame: 고객 데이터
    """
    np.random.seed(seed)
    random.seed(seed)
    
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    
    # 고객 속성 생성
    data = {
        'customer_id': customer_ids,
        'age': np.random.randint(20, 70, n_customers),
        'gender': np.random.choice(['M', 'F'], n_customers),
        'region': np.random.choice(['서울', '경기', '부산', '인천', '대구', '기타'], n_customers, p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.1]),
        'customer_type': np.random.choice(['개인', '기업'], n_customers, p=[0.7, 0.3]),
        'join_date': [datetime.now() - timedelta(days=np.random.randint(1, 1095)) for _ in range(n_customers)],
        'total_spent': np.random.lognormal(mean=10, sigma=1.5, size=n_customers).round(2),
        'total_orders': np.random.poisson(lam=15, size=n_customers),
        'avg_order_value': np.random.lognormal(mean=8, sigma=1, size=n_customers).round(2),
        'last_order_days': np.random.randint(0, 180, n_customers),
        'support_tickets': np.random.poisson(lam=2, size=n_customers),
        'payment_method': np.random.choice(['신용카드', '계좌이체', '페이', '기타'], n_customers, p=[0.4, 0.3, 0.2, 0.1]),
        'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], n_customers, p=[0.5, 0.3, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    # 해지 여부 생성 (실제 모델이 예측할 타겟)
    # 여러 요인을 조합하여 해지 확률 계산
    churn_prob = (
        (df['last_order_days'] > 90) * 0.3 +
        (df['support_tickets'] > 5) * 0.2 +
        (df['total_orders'] < 5) * 0.2 +
        (df['avg_order_value'] < df['avg_order_value'].quantile(0.25)) * 0.15 +
        np.random.random(n_customers) * 0.15
    )
    df['churn'] = (churn_prob > 0.5).astype(int)
    df['churn_probability'] = churn_prob
    
    return df


def generate_seller_data(n_sellers=200, seed=42):
    """
    판매자 데이터 생성
    
    Args:
        n_sellers: 생성할 판매자 수
        seed: 랜덤 시드
    
    Returns:
        DataFrame: 판매자 데이터
    """
    np.random.seed(seed)
    random.seed(seed)
    
    seller_ids = [f"SELL_{i:04d}" for i in range(1, n_sellers + 1)]
    
    data = {
        'seller_id': seller_ids,
        'seller_name': [f"판매자{i}" for i in range(1, n_sellers + 1)],
        'category': np.random.choice(['IT개발', '디자인', '마케팅', '번역', '기타'], n_sellers, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        'rating': np.random.normal(4.2, 0.5, n_sellers).clip(1, 5).round(2),
        'total_projects': np.random.poisson(lam=50, size=n_sellers),
        'join_date': [datetime.now() - timedelta(days=np.random.randint(1, 1825)) for _ in range(n_sellers)],
        'response_time_hours': np.random.exponential(scale=12, size=n_sellers).round(1),
        'completion_rate': np.random.normal(0.95, 0.05, n_sellers).clip(0.7, 1.0).round(3),
        'revenue': np.random.lognormal(mean=12, sigma=1.2, size=n_sellers).round(2),
    }
    
    return pd.DataFrame(data)


def generate_transaction_data(n_transactions=5000, customer_df=None, seller_df=None, seed=42):
    """
    거래 데이터 생성
    
    Args:
        n_transactions: 생성할 거래 수
        customer_df: 고객 데이터프레임
        seller_df: 판매자 데이터프레임
        seed: 랜덤 시드
    
    Returns:
        DataFrame: 거래 데이터
    """
    np.random.seed(seed)
    random.seed(seed)
    
    if customer_df is None:
        customer_df = generate_customer_data()
    if seller_df is None:
        seller_df = generate_seller_data()
    
    transaction_ids = [f"TXN_{i:06d}" for i in range(1, n_transactions + 1)]
    
    # 고객과 판매자 무작위 선택
    customer_ids = np.random.choice(customer_df['customer_id'].values, n_transactions)
    seller_ids = np.random.choice(seller_df['seller_id'].values, n_transactions)
    
    # 거래 날짜 생성 (최근 1년)
    start_date = datetime.now() - timedelta(days=365)
    transaction_dates = [
        start_date + timedelta(days=np.random.randint(0, 365))
        for _ in range(n_transactions)
    ]
    
    data = {
        'transaction_id': transaction_ids,
        'customer_id': customer_ids,
        'seller_id': seller_ids,
        'transaction_date': transaction_dates,
        'amount': np.random.lognormal(mean=9, sigma=1.2, size=n_transactions).round(2),
        'project_type': np.random.choice(['웹개발', '앱개발', '디자인', '마케팅', '기타'], n_transactions, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'status': np.random.choice(['완료', '진행중', '취소', '환불'], n_transactions, p=[0.7, 0.15, 0.1, 0.05]),
        'payment_status': np.random.choice(['완료', '대기', '실패'], n_transactions, p=[0.85, 0.1, 0.05]),
        'duration_days': np.random.exponential(scale=14, size=n_transactions).round(0).astype(int),
    }
    
    return pd.DataFrame(data)


def generate_all_sample_data(n_customers=1000, n_sellers=200, n_transactions=5000, seed=42):
    """
    모든 샘플 데이터 생성
    
    Args:
        n_customers: 고객 수
        n_sellers: 판매자 수
        n_transactions: 거래 수
        seed: 랜덤 시드
    
    Returns:
        tuple: (customer_df, seller_df, transaction_df)
    """
    customer_df = generate_customer_data(n_customers, seed)
    seller_df = generate_seller_data(n_sellers, seed)
    transaction_df = generate_transaction_data(n_transactions, customer_df, seller_df, seed)
    
    return customer_df, seller_df, transaction_df


if __name__ == "__main__":
    # 테스트 실행
    print("샘플 데이터 생성 중...")
    customer_df, seller_df, transaction_df = generate_all_sample_data()
    
    print(f"\n고객 데이터: {customer_df.shape}")
    print(customer_df.head())
    
    print(f"\n판매자 데이터: {seller_df.shape}")
    print(seller_df.head())
    
    print(f"\n거래 데이터: {transaction_df.shape}")
    print(transaction_df.head())
    
    print(f"\n해지율: {customer_df['churn'].mean():.2%}")


