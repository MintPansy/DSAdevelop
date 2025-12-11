"""
데이터 검증 및 시각화 스크립트
생성된 더미데이터의 품질을 검증하고 시각화합니다.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False


def load_data(data_dir="data"):
    """CSV 파일에서 데이터 로드"""
    data_path = Path(data_dir)
    
    transactions_df = pd.read_csv(data_path / "transactions.csv", encoding='utf-8-sig')
    customers_df = pd.read_csv(data_path / "customers.csv", encoding='utf-8-sig')
    predictions_df = pd.read_csv(data_path / "predictions.csv", encoding='utf-8-sig')
    
    # 날짜 컬럼 변환
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    transactions_df['cancellation_date'] = pd.to_datetime(transactions_df['cancellation_date'], errors='coerce')
    customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'])
    
    return transactions_df, customers_df, predictions_df


def create_visualizations(transactions_df, customers_df, predictions_df, output_dir="data/validation_report"):
    """
    데이터 시각화 생성
    
    Args:
        transactions_df: 거래 데이터프레임
        customers_df: 고객 데이터프레임
        predictions_df: 예측 데이터프레임
        output_dir: 출력 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n시각화 생성 중...")
    
    # 1. 거래금액 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(transactions_df['sales_amount'] / 1000000, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('거래금액 (백만원)')
    plt.ylabel('빈도')
    plt.title('거래금액 분포')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "1_sales_amount_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 거래금액 분포 히스토그램 저장")
    
    # 2. 수정요청 횟수 분포
    plt.figure(figsize=(10, 6))
    plt.hist(transactions_df['modification_count'], bins=11, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('수정요청 횟수')
    plt.ylabel('빈도')
    plt.title('수정요청 횟수 분포')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "2_modification_count_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 수정요청 횟수 분포 히스토그램 저장")
    
    # 3. 평점 분포
    plt.figure(figsize=(10, 6))
    plt.hist(transactions_df['service_rating'], bins=20, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('평점')
    plt.ylabel('빈도')
    plt.title('서비스 평점 분포')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "3_service_rating_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 평점 분포 히스토그램 저장")
    
    # 4. 수정요청 vs 거래취소 산점도
    plt.figure(figsize=(10, 6))
    canceled = transactions_df[transactions_df['transaction_canceled'] == 1]
    not_canceled = transactions_df[transactions_df['transaction_canceled'] == 0]
    plt.scatter(not_canceled['modification_count'], not_canceled['service_rating'], 
                alpha=0.3, label='정상 거래', s=20)
    plt.scatter(canceled['modification_count'], canceled['service_rating'], 
                alpha=0.7, label='취소 거래', s=50, color='red')
    plt.xlabel('수정요청 횟수')
    plt.ylabel('평점')
    plt.title('수정요청 횟수 vs 평점 (거래취소 여부)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "4_modification_vs_rating_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 수정요청 vs 평점 산점도 저장")
    
    # 5. 상관관계 히트맵
    plt.figure(figsize=(12, 10))
    numeric_cols = ['sales_amount', 'modification_count', 'additional_payment', 
                    'service_rating', 'transaction_canceled']
    corr_matrix = transactions_df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('거래 데이터 상관관계 매트릭스')
    plt.tight_layout()
    plt.savefig(output_path / "5_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 상관관계 히트맵 저장")
    
    # 6. 세그먼트별 통계 박스플롯
    plt.figure(figsize=(12, 6))
    customers_with_predictions = customers_df.merge(predictions_df, on='customer_id')
    segment_order = ['일반', '프리미엄', 'VIP']
    customers_with_predictions['customer_segment'] = pd.Categorical(
        customers_with_predictions['customer_segment'], categories=segment_order, ordered=True
    )
    customers_with_predictions = customers_with_predictions.sort_values('customer_segment')
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=customers_with_predictions, x='customer_segment', y='total_purchase_amount' / 1000000)
    plt.xlabel('고객 세그먼트')
    plt.ylabel('총 구매금액 (백만원)')
    plt.title('세그먼트별 총 구매금액')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=customers_with_predictions, x='customer_segment', y='churn_probability')
    plt.xlabel('고객 세그먼트')
    plt.ylabel('해지 확률')
    plt.title('세그먼트별 해지 확률')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / "6_segment_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 세그먼트별 비교 박스플롯 저장")
    
    # 7. 리스크 레벨 분포
    plt.figure(figsize=(10, 6))
    risk_counts = predictions_df['risk_level'].value_counts()
    colors = ['green', 'orange', 'red']
    plt.bar(risk_counts.index, risk_counts.values, color=colors[:len(risk_counts)], alpha=0.7, edgecolor='black')
    plt.xlabel('리스크 레벨')
    plt.ylabel('고객 수')
    plt.title('리스크 레벨 분포')
    for i, (level, count) in enumerate(risk_counts.items()):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(output_path / "7_risk_level_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 리스크 레벨 분포 저장")
    
    # 8. 시간별 거래 추이
    plt.figure(figsize=(14, 6))
    transactions_df['transaction_month'] = transactions_df['transaction_date'].dt.to_period('M').astype(str)
    monthly_stats = transactions_df.groupby('transaction_month').agg({
        'transaction_id': 'count',
        'transaction_canceled': 'mean'
    }).reset_index()
    
    plt.subplot(1, 2, 1)
    plt.plot(monthly_stats['transaction_month'], monthly_stats['transaction_id'], marker='o')
    plt.xlabel('월')
    plt.ylabel('거래 수')
    plt.title('월별 거래 수 추이')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(monthly_stats['transaction_month'], monthly_stats['transaction_canceled'] * 100, 
             marker='o', color='red')
    plt.xlabel('월')
    plt.ylabel('거래취소율 (%)')
    plt.title('월별 거래취소율 추이')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "8_monthly_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 시간별 추이 그래프 저장")
    
    print(f"\n모든 시각화 파일이 저장되었습니다: {output_path}")


def generate_validation_report(transactions_df, customers_df, predictions_df, output_file="data/validation_report.txt"):
    """
    검증 리포트 텍스트 파일 생성
    
    Args:
        transactions_df: 거래 데이터프레임
        customers_df: 고객 데이터프레임
        predictions_df: 예측 데이터프레임
        output_file: 출력 파일 경로
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("더미데이터 검증 리포트\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 기본 정보
        f.write("[1] 기본 통계\n")
        f.write("-" * 80 + "\n")
        f.write(f"거래 데이터: {transactions_df.shape[0]:,}행, {transactions_df.shape[1]}열\n")
        f.write(f"고객 데이터: {customers_df.shape[0]:,}행, {customers_df.shape[1]}열\n")
        f.write(f"예측 데이터: {predictions_df.shape[0]:,}행, {predictions_df.shape[1]}열\n\n")
        
        # 2. 거래 데이터 통계
        f.write("[2] 거래 데이터 통계\n")
        f.write("-" * 80 + "\n")
        f.write(f"거래금액:\n")
        f.write(f"  평균: {transactions_df['sales_amount'].mean():,.0f}원\n")
        f.write(f"  표준편차: {transactions_df['sales_amount'].std():,.0f}원\n")
        f.write(f"  최소: {transactions_df['sales_amount'].min():,.0f}원\n")
        f.write(f"  최대: {transactions_df['sales_amount'].max():,.0f}원\n")
        f.write(f"  중앙값: {transactions_df['sales_amount'].median():,.0f}원\n\n")
        
        f.write(f"수정요청 횟수:\n")
        f.write(f"  평균: {transactions_df['modification_count'].mean():.2f}회\n")
        f.write(f"  최대: {transactions_df['modification_count'].max()}회\n")
        f.write(f"  분포: {transactions_df['modification_count'].value_counts().sort_index().to_dict()}\n\n")
        
        f.write(f"거래취소율: {transactions_df['transaction_canceled'].mean():.4f} ({transactions_df['transaction_canceled'].mean()*100:.2f}%)\n")
        f.write(f"  취소 거래 수: {transactions_df['transaction_canceled'].sum():,}건\n\n")
        
        # 3. 상관관계
        f.write("[3] 상관관계 분석\n")
        f.write("-" * 80 + "\n")
        numeric_cols = ['sales_amount', 'modification_count', 'additional_payment', 
                       'service_rating', 'transaction_canceled']
        corr_matrix = transactions_df[numeric_cols].corr()
        f.write(corr_matrix.to_string())
        f.write("\n\n")
        
        # 4. 세그먼트별 통계
        f.write("[4] 세그먼트별 통계\n")
        f.write("-" * 80 + "\n")
        segment_stats = customers_df.groupby('customer_segment').agg({
            'total_purchase_amount': ['mean', 'count', 'sum'],
            'total_modification_count': 'mean',
            'age': 'mean'
        })
        f.write(segment_stats.to_string())
        f.write("\n\n")
        
        # 5. 해지 확률 분포
        f.write("[5] 해지 확률 분포\n")
        f.write("-" * 80 + "\n")
        f.write(f"평균 해지 확률: {predictions_df['churn_probability'].mean():.4f}\n")
        f.write(f"표준편차: {predictions_df['churn_probability'].std():.4f}\n")
        f.write(f"최소: {predictions_df['churn_probability'].min():.4f}\n")
        f.write(f"최대: {predictions_df['churn_probability'].max():.4f}\n\n")
        
        f.write("리스크 레벨 분포:\n")
        f.write(predictions_df['risk_level'].value_counts().to_string())
        f.write("\n\n")
        
        # 6. 검증 체크리스트
        f.write("[6] 검증 체크리스트\n")
        f.write("-" * 80 + "\n")
        
        checks = []
        # 행 수 확인
        checks.append(("행 수", transactions_df.shape[0] == 10000, f"{transactions_df.shape[0]}행"))
        # 거래금액 범위
        amount_ok = (transactions_df['sales_amount'].min() >= 1000000 and 
                    transactions_df['sales_amount'].max() <= 30000000)
        checks.append(("거래금액 범위", amount_ok, 
                      f"{transactions_df['sales_amount'].min():,.0f}~{transactions_df['sales_amount'].max():,.0f}원"))
        # 상관관계
        corr_mod = transactions_df['modification_count'].corr(transactions_df['transaction_canceled'])
        checks.append(("수정요청-취소 상관관계", 0.5 <= corr_mod <= 0.7, f"{corr_mod:.3f}"))
        # 취소율
        cancel_rate = transactions_df['transaction_canceled'].mean()
        checks.append(("거래취소율", 0.01 <= cancel_rate <= 0.03, f"{cancel_rate:.4f}"))
        # 결측치
        no_missing = transactions_df.isnull().sum().sum() == 0
        checks.append(("결측치 없음", no_missing, f"{transactions_df.isnull().sum().sum()}개"))
        
        for check_name, result, value in checks:
            status = "✓" if result else "✗"
            f.write(f"{status} {check_name}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("리포트 생성 완료\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n검증 리포트 저장 완료: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("데이터 검증 및 시각화 시작")
    print("=" * 60)
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    transactions_df, customers_df, predictions_df = load_data()
    print("✓ 데이터 로드 완료")
    
    # 검증 리포트 생성
    generate_validation_report(transactions_df, customers_df, predictions_df)
    
    # 시각화 생성
    create_visualizations(transactions_df, customers_df, predictions_df)
    
    print("\n✅ 모든 검증 및 시각화 작업 완료!")

