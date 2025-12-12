"""
IT 아웃소싱 플랫폼 고객 해지예측 Streamlit 대시보드
Decision Tree 모델을 활용한 실시간 리스크 스코어 표시
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent))

from data.sample_data import generate_all_sample_data
from models.predictor import ChurnPredictor
import os
from utils.visualization import (
    create_risk_score_gauge,
    create_churn_distribution_chart,
    create_risk_score_distribution,
    create_feature_importance_chart,
    create_customer_segmentation_chart,
    create_time_series_churn,
    create_correlation_heatmap
)
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="고객 해지예측 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 - Professional + 친화적 색상 팔레트
st.markdown("""
    <style>
    /* Hero 섹션 스타일 */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #f0f0f0;
        text-align: center;
        font-weight: 300;
    }
    
    /* 메트릭 카드 스타일 */
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* 탭 네비게이션 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* 메인 컨텐츠 스타일 */
    .main .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """
    샘플 데이터 로드 (캐싱)
    CSV 파일이 있으면 CSV에서 로드, 없으면 더미데이터 생성
    """
    data_path = Path("data")
    
    # CSV 파일 존재 여부 확인
    transactions_csv = data_path / "transactions.csv"
    customers_csv = data_path / "customers.csv"
    predictions_csv = data_path / "predictions.csv"
    
    if transactions_csv.exists() and customers_csv.exists() and predictions_csv.exists():
        # CSV 파일에서 로드
        try:
            transactions_df = pd.read_csv(transactions_csv, encoding='utf-8-sig')
            customers_df = pd.read_csv(customers_csv, encoding='utf-8-sig')
            predictions_df = pd.read_csv(predictions_csv, encoding='utf-8-sig')
            
            # 날짜 컬럼 변환
            transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
            transactions_df['cancellation_date'] = pd.to_datetime(
                transactions_df['cancellation_date'], errors='coerce'
            )
            customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'])
            
            # seller_df는 더미데이터 (거래 데이터에서 추출)
            seller_df = pd.DataFrame({
                'seller_id': transactions_df['customer_id'].unique()[:200] if len(transactions_df) > 0 else []
            })
            
            return customers_df, seller_df, transactions_df, predictions_df
        except Exception as e:
            st.warning(f"CSV 파일 로드 중 오류 발생: {e}. 더미데이터를 생성합니다.")
    
    # CSV 파일이 없으면 기존 방식으로 생성
    customer_df, seller_df, transaction_df = generate_all_sample_data(
        n_customers=1000,
        n_sellers=200,
        n_transactions=5000
    )
    
    # predictions_df 생성 (기존 방식)
    predictions_df = pd.DataFrame({
        'customer_id': customer_df['customer_id'],
        'churn_probability': customer_df.get('churn_probability', np.random.random(len(customer_df))),
        'risk_level': pd.cut(
            customer_df.get('churn_probability', np.random.random(len(customer_df))),
            bins=[0, 0.3, 0.7, 1.0],
            labels=['낮음', '중간', '높음']
        )
    })
    
    return customer_df, seller_df, transaction_df, predictions_df


def calculate_recent_churn_rate(transaction_df, days=7):
    """
    최근 N일 평균 해지율 계산 (거래 취소율)
    
    Args:
        transaction_df: 거래 데이터프레임
        days: 최근 며칠간 (기본 7일)
    
    Returns:
        float: 최근 해지율 (%)
    """
    from datetime import datetime, timedelta
    
    # 최근 N일 이내 거래 필터링
    # 입력 데이터프레임을 수정하지 않도록 복사본 생성
    transaction_df_copy = transaction_df.copy()
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # transaction_date가 이미 datetime이 아닐 수 있으므로 변환
    if not pd.api.types.is_datetime64_any_dtype(transaction_df_copy['transaction_date']):
        transaction_df_copy['transaction_date'] = pd.to_datetime(transaction_df_copy['transaction_date'])
    
    recent_transactions = transaction_df_copy[
        transaction_df_copy['transaction_date'] >= cutoff_date
    ]
    
    # 최근 거래 중 취소율 계산
    if len(recent_transactions) > 0 and 'transaction_canceled' in recent_transactions.columns:
        return recent_transactions['transaction_canceled'].mean() * 100
    
    # 대체: 전체 거래 취소율
    if 'transaction_canceled' in transaction_df_copy.columns:
        return transaction_df_copy['transaction_canceled'].mean() * 100
    
    return 0.0


@st.cache_resource
def load_predictor():
    """모델 로드 (캐싱)"""
    return ChurnPredictor()


def main():
    """메인 함수"""
    # Hero 섹션
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">🔴 IT 아웃소싱 고객 해지예측 대시보드</div>
            <div class="hero-subtitle">데이터 기반 고객 리스크 모니터링 시스템</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 데이터 로드
    with st.spinner("데이터 로딩 중..."):
        customer_df, seller_df, transaction_df, predictions_df = load_sample_data()
        predictor = load_predictor()
    
    # 예측 데이터와 고객 데이터 병합
    if 'customer_id' in predictions_df.columns:
        customer_df = customer_df.merge(predictions_df, on='customer_id', how='left')
        # risk_score는 churn_probability * 100으로 계산
        if 'churn_probability' in customer_df.columns:
            customer_df['risk_score'] = (customer_df['churn_probability'] * 100).round(2)
            customer_df['predicted_churn'] = (customer_df['churn_probability'] > 0.5).astype(int)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 데이터 필터
        st.subheader("📊 필터")
        selected_regions = st.multiselect(
            "지역 선택",
            options=sorted(customer_df['region'].unique()) if 'region' in customer_df.columns else [],
            default=sorted(customer_df['region'].unique()) if 'region' in customer_df.columns else []
        )
        
        # customer_segment 또는 subscription_type
        segment_col = 'customer_segment' if 'customer_segment' in customer_df.columns else 'subscription_type'
        if segment_col in customer_df.columns:
            selected_segments = st.multiselect(
                "고객 세그먼트" if segment_col == 'customer_segment' else "구독 유형",
                options=sorted(customer_df[segment_col].unique()),
                default=sorted(customer_df[segment_col].unique())
            )
        else:
            selected_segments = []
        
        risk_threshold = st.slider(
            "리스크 스코어 임계값",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )
        
        st.divider()
        st.markdown("**💡 팁**: 필터를 조정하여 특정 고객 그룹을 분석할 수 있습니다.")
    
    # 필터 적용
    filter_conditions = []
    if selected_regions and 'region' in customer_df.columns:
        filter_conditions.append(customer_df['region'].isin(selected_regions))
    if selected_segments and segment_col in customer_df.columns:
        filter_conditions.append(customer_df[segment_col].isin(selected_segments))
    
    if filter_conditions:
        filtered_df = customer_df[np.logical_and.reduce(filter_conditions)].copy()
    else:
        filtered_df = customer_df.copy()
    
    # 상단 메트릭 요약
    show_metrics_summary(filtered_df, transaction_df)
    
    # 탭 네비게이션
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 개별 고객 조회",
        "📊 세그먼트 분석",
        "🧪 A/B 테스트",
        "📦 배치 분석"
    ])
    
    with tab1:
        show_customer_detail(filtered_df, transaction_df, predictor)
    
    with tab2:
        show_segment_analysis(filtered_df, predictor)
    
    with tab3:
        show_ab_test(filtered_df, predictor)
    
    with tab4:
        show_batch_analysis(filtered_df, customer_df, transaction_df, predictor)


def show_metrics_summary(df, transaction_df):
    """상단 메트릭 요약 섹션"""
    st.markdown("### 📊 주요 지표")
    
    # 메트릭 계산
    avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0
    high_risk_count = (df['risk_score'] >= 70).sum() if 'risk_score' in df.columns else 0
    recent_churn_rate = calculate_recent_churn_rate(transaction_df, days=7)
    
    # 메트릭 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="평균 해지 위험도",
            value=f"{avg_risk:.1f}%",
            delta=f"{avg_risk - 50:.1f}%p" if avg_risk > 0 else None,
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="고위험 고객 수",
            value=f"{high_risk_count:,}명",
            delta=f"{high_risk_count - (len(df) * 0.1):.0f}명" if high_risk_count > 0 else None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="최근 7일 평균 해지율",
            value=f"{recent_churn_rate:.2f}%",
            delta=f"{recent_churn_rate - 2.0:.2f}%p" if recent_churn_rate > 0 else None,
            delta_color="inverse"
        )
    
    st.divider()


def show_segment_analysis(df, predictor):
    """세그먼트 분석 페이지"""
    st.header("📊 세그먼트 분석")
    
    # 세그먼트 선택
    segment_type = st.radio(
        "세그먼트 기준",
        ["구독 유형", "지역", "고객 유형"],
        horizontal=True
    )
    
    # 세그먼트별 분석
    if segment_type == "구독 유형":
        # customer_segment 또는 subscription_type
        segment_col = 'customer_segment' if 'customer_segment' in df.columns else 'subscription_type'
    elif segment_type == "지역":
        segment_col = 'region'
    else:
        segment_col = 'customer_type' if 'customer_type' in df.columns else 'customer_segment'
    
    # 세그먼트별 통계
    segment_stats = df.groupby(segment_col).agg({
        'risk_score': ['mean', 'count'],
        'predicted_churn': 'mean' if 'predicted_churn' in df.columns else 'count'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{segment_type}별 해지율")
        fig = create_customer_segmentation_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"{segment_type}별 평균 리스크 스코어")
        if 'risk_score' in df.columns:
            segment_risk = df.groupby(segment_col)['risk_score'].mean().sort_values(ascending=False)
            fig = go.Figure(data=[
                go.Bar(
                    x=segment_risk.index,
                    y=segment_risk.values,
                    marker_color='#667eea',
                    text=segment_risk.values.round(1),
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title=f"{segment_type}별 평균 리스크 스코어",
                xaxis_title=segment_type,
                yaxis_title="평균 리스크 스코어",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 세그먼트별 상세 통계 테이블
    st.subheader("세그먼트별 상세 통계")
    if 'risk_score' in df.columns:
        # 사용 가능한 컬럼만 집계
        agg_dict = {
            'risk_score': ['mean', 'std', 'min', 'max'],
            'predicted_churn': 'sum' if 'predicted_churn' in df.columns else 'count'
        }
        if 'total_purchase_amount' in df.columns:
            agg_dict['total_purchase_amount'] = 'mean'
        elif 'total_spent' in df.columns:
            agg_dict['total_spent'] = 'mean'
        if 'total_modification_count' in df.columns:
            agg_dict['total_modification_count'] = 'mean'
        
        segment_detail = df.groupby(segment_col).agg(agg_dict).round(2)
        st.dataframe(segment_detail, use_container_width=True)
    
    # 특성 중요도
    st.subheader("특성 중요도")
    feature_importance = predictor.get_feature_importance()
    if feature_importance:
        fig = create_feature_importance_chart(feature_importance)
        st.plotly_chart(fig, use_container_width=True)


def show_ab_test(df, predictor):
    """A/B 테스트 시뮬레이터 페이지"""
    st.header("🧪 A/B 테스트 시뮬레이터")
    
    st.info("💡 이 페이지에서는 다양한 개입 전략의 효과를 시뮬레이션할 수 있습니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("테스트 설정")
        intervention_type = st.selectbox(
            "개입 유형",
            ["프로모션 제공", "고객 서비스 개선", "할인 쿠폰", "프리미엄 업그레이드"]
        )
        
        target_segment = st.selectbox(
            "타겟 세그먼트",
            ["전체", "고위험 고객 (70점 이상)", "중위험 고객 (30-70점)", "저위험 고객 (30점 미만)"]
        )
        
        intervention_rate = st.slider(
            "개입 효과 (해지율 감소율)",
            min_value=0,
            max_value=50,
            value=20,
            step=5,
            help="개입으로 인한 해지율 감소 비율 (%)"
        )
    
    with col2:
        st.subheader("예상 결과")
        
        # 타겟 고객 필터링
        if target_segment == "전체":
            target_df = df
        elif target_segment == "고위험 고객 (70점 이상)":
            target_df = df[df['risk_score'] >= 70] if 'risk_score' in df.columns else df
        elif target_segment == "중위험 고객 (30-70점)":
            target_df = df[(df['risk_score'] >= 30) & (df['risk_score'] < 70)] if 'risk_score' in df.columns else df
        else:
            target_df = df[df['risk_score'] < 30] if 'risk_score' in df.columns else df
        
        if len(target_df) > 0 and 'predicted_churn' in target_df.columns:
            current_churn_rate = target_df['predicted_churn'].mean() * 100
            expected_churn_rate = current_churn_rate * (1 - intervention_rate / 100)
            reduction = current_churn_rate - expected_churn_rate
            
            st.metric("현재 해지율", f"{current_churn_rate:.2f}%")
            st.metric("예상 해지율", f"{expected_churn_rate:.2f}%", 
                     delta=f"-{reduction:.2f}%p", delta_color="normal")
            st.metric("타겟 고객 수", f"{len(target_df):,}명")
            
            # ROI 계산
            avg_customer_value = target_df['total_spent'].mean() if 'total_spent' in target_df.columns else 0
            saved_customers = len(target_df) * (reduction / 100)
            estimated_value = saved_customers * avg_customer_value
            
            st.metric("예상 절감 고객 수", f"{saved_customers:.0f}명")
            st.metric("예상 가치 보존", f"{estimated_value:,.0f}원")
        else:
            st.warning("타겟 고객 데이터가 없습니다.")
    
    # 시각화
    if len(target_df) > 0 and 'risk_score' in target_df.columns:
        st.subheader("리스크 분포 비교")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**개입 전**")
            fig_before = create_risk_score_distribution(target_df)
            st.plotly_chart(fig_before, use_container_width=True)
        
        with col2:
            st.write("**개입 후 (예상)**")
            # 개입 후 리스크 스코어 시뮬레이션
            target_df_after = target_df.copy()
            target_df_after['risk_score'] = target_df_after['risk_score'] * (1 - intervention_rate / 100)
            fig_after = create_risk_score_distribution(target_df_after)
            st.plotly_chart(fig_after, use_container_width=True)


def show_batch_analysis(df, customer_df, transaction_df, predictor):
    """배치 분석 페이지"""
    st.header("📦 배치 분석")
    
    # 분석 옵션
    analysis_type = st.selectbox(
        "분석 유형",
        ["전체 현황", "시간별 추이", "상관관계 분석", "원본 데이터"]
    )
    
    if analysis_type == "전체 현황":
        st.subheader("📈 전체 현황 대시보드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**해지 분포**")
            if 'predicted_churn' in df.columns:
                chart_df = df.copy()
                chart_df['churn'] = chart_df['predicted_churn']
                fig = create_churn_distribution_chart(chart_df)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**리스크 스코어 분포**")
            if 'risk_score' in df.columns:
                fig = create_risk_score_distribution(df)
                st.plotly_chart(fig, use_container_width=True)
        
        # 고위험 고객 리스트
        st.subheader("⚠️ 고위험 고객 리스트")
        if 'risk_score' in df.columns:
            high_risk_customers = df[df['risk_score'] >= 70].sort_values('risk_score', ascending=False)
            
            if len(high_risk_customers) > 0:
                display_cols = ['customer_id', 'region', 'subscription_type', 'total_orders', 
                              'last_order_days', 'risk_score', 'churn_probability']
                available_cols = [col for col in display_cols if col in high_risk_customers.columns]
                
                st.dataframe(
                    high_risk_customers[available_cols].head(50),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("고위험 고객이 없습니다.")
    
    elif analysis_type == "시간별 추이":
        st.subheader("📅 시간별 해지율 추이")
        fig = create_time_series_churn(customer_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("날짜 데이터가 없습니다.")
    
    elif analysis_type == "상관관계 분석":
        st.subheader("🔗 특성 간 상관관계")
        fig = create_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.subheader("📋 원본 데이터")
        
        data_type = st.selectbox("데이터 유형", ["고객 데이터", "거래 데이터"])
        
        if data_type == "고객 데이터":
            st.dataframe(customer_df, use_container_width=True)
        else:
            st.dataframe(transaction_df, use_container_width=True)


def show_customer_detail(df, transaction_df, predictor):
    """개별 고객 조회 페이지"""
    st.header("🔍 개별 고객 조회")
    
    # 고객 선택
    customer_ids = df['customer_id'].tolist()
    selected_id = st.selectbox("고객 ID 선택", customer_ids, index=0)
    
    if selected_id:
        customer = df[df['customer_id'] == selected_id].iloc[0]
        
        # 해당 고객의 거래 이력
        customer_transactions = transaction_df[transaction_df['customer_id'] == selected_id].copy()
        
        # 상단 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_transactions = len(customer_transactions)
            st.metric("총 거래 수", f"{total_transactions}건")
        
        with col2:
            total_amount = customer_transactions['sales_amount'].sum() if len(customer_transactions) > 0 else 0
            st.metric("총 거래금액", f"{total_amount:,.0f}원")
        
        with col3:
            canceled_count = customer_transactions['transaction_canceled'].sum() if len(customer_transactions) > 0 else 0
            st.metric("취소 거래", f"{canceled_count}건", delta=f"-{canceled_count}건" if canceled_count > 0 else None)
        
        with col4:
            avg_rating = customer_transactions['service_rating'].mean() if len(customer_transactions) > 0 and 'service_rating' in customer_transactions.columns else 0
            st.metric("평균 평점", f"{avg_rating:.1f}" if avg_rating > 0 else "N/A")
        
        st.divider()
        
        # 고객 정보 및 리스크 분석
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 고객 기본 정보")
            info_items = []
            
            # 기본 정보
            if 'customer_id' in customer:
                info_items.append(("고객 ID", customer['customer_id']))
            if 'age' in customer:
                info_items.append(("나이", f"{int(customer['age'])}세"))
            if 'region' in customer:
                info_items.append(("지역", customer['region']))
            if 'customer_segment' in customer:
                info_items.append(("고객 세그먼트", customer['customer_segment']))
            elif 'subscription_type' in customer:
                info_items.append(("구독 유형", customer['subscription_type']))
            if 'registration_date' in customer:
                reg_date = pd.to_datetime(customer['registration_date'])
                days_since = (datetime.now() - reg_date).days
                info_items.append(("가입일", f"{reg_date.strftime('%Y-%m-%d')} ({days_since}일 전)"))
            
            # 구매 통계
            if 'total_purchase_amount' in customer:
                info_items.append(("총 구매금액", f"{customer['total_purchase_amount']:,.0f}원"))
            elif 'total_spent' in customer:
                info_items.append(("총 구매금액", f"{customer['total_spent']:,.0f}원"))
            
            if 'total_modification_count' in customer:
                info_items.append(("총 수정요청", f"{int(customer['total_modification_count'])}회"))
            if 'total_additional_payment' in customer:
                info_items.append(("총 추가결제", f"{customer['total_additional_payment']:,.0f}원"))
            
            for key, value in info_items:
                st.write(f"**{key}**: {value}")
        
        with col2:
            st.subheader("⚠️ 리스크 분석")
            if 'risk_score' in customer:
                risk_score = float(customer['risk_score'])
                fig = create_risk_score_gauge(risk_score)
                st.plotly_chart(fig, use_container_width=True)
                
                if 'churn_probability' in customer:
                    churn_prob = float(customer['churn_probability'])
                    st.metric("해지 확률", f"{churn_prob*100:.2f}%")
                
                if 'risk_level' in customer:
                    risk_level = customer['risk_level']
                    risk_color = {'높음': '🔴', '중간': '🟡', '낮음': '🟢'}.get(risk_level, '⚪')
                    st.metric("리스크 레벨", f"{risk_color} {risk_level}")
                
                if 'predicted_churn' in customer:
                    predicted = int(customer['predicted_churn'])
                    st.metric("예상 해지 여부", "해지 예상" if predicted == 1 else "유지 예상")
            else:
                st.info("리스크 스코어 정보가 없습니다.")
        
        st.divider()
        
        # 거래 이력
        st.subheader("📊 거래 이력")
        
        if len(customer_transactions) > 0:
            # 거래 이력 요약
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**거래 통계**")
                st.write(f"- 평균 거래금액: {customer_transactions['sales_amount'].mean():,.0f}원")
                st.write(f"- 평균 수정요청: {customer_transactions['modification_count'].mean():.1f}회")
                st.write(f"- 평균 추가결제: {customer_transactions['additional_payment'].mean():,.0f}원")
            
            with col2:
                st.write("**서비스 카테고리**")
                if 'service_category' in customer_transactions.columns:
                    category_counts = customer_transactions['service_category'].value_counts()
                    for cat, count in category_counts.items():
                        st.write(f"- {cat}: {count}건")
            
            # 거래 이력 테이블
            st.write("**최근 거래 내역**")
            display_cols = ['transaction_date', 'sales_amount', 'service_category', 
                          'modification_count', 'service_rating', 'transaction_canceled']
            available_cols = [col for col in display_cols if col in customer_transactions.columns]
            
            # 날짜 순으로 정렬
            if 'transaction_date' in customer_transactions.columns:
                customer_transactions_sorted = customer_transactions.sort_values('transaction_date', ascending=False)
            else:
                customer_transactions_sorted = customer_transactions
            
            st.dataframe(
                customer_transactions_sorted[available_cols].head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # 거래 추이 차트
            if 'transaction_date' in customer_transactions.columns and len(customer_transactions) > 1:
                st.subheader("거래 추이")
                fig = go.Figure()
                
                # 거래금액 추이
                customer_transactions_sorted = customer_transactions.sort_values('transaction_date')
                fig.add_trace(go.Scatter(
                    x=customer_transactions_sorted['transaction_date'],
                    y=customer_transactions_sorted['sales_amount'],
                    mode='lines+markers',
                    name='거래금액',
                    line=dict(color='#667eea', width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="거래금액 추이",
                    xaxis_title="날짜",
                    yaxis_title="거래금액 (원)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("거래 이력이 없습니다.")




if __name__ == "__main__":
    main()

# app.py에 추가하기

import streamlit as st
import pandas as pd
from datetime import datetime

# 1. Streamlit 설정 (자동 새로고침)
st.set_page_config(
    page_title="고객 해지예측 대시보드",
    initial_sidebar_state="expanded",
)

# 2. 자동 새로고침 설정 (매 5분마다 데이터 갱신)
st.markdown("""
    <meta http-equiv="refresh" content="300">
""", unsafe_allow_html=True)

# 3. 데이터 로드 함수 (캐싱 시간 제한)
@st.cache_data(ttl=300)  # 300초(5분) 후 캐시 무효화
def load_data():
    customers = pd.read_csv('data/customers.csv')
    transactions = pd.read_csv('data/transactions.csv')
    return customers, transactions

# 4. 마지막 업데이트 시간 표시
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔴 IT 아웃소싱 고객 해지예측 대시보드")
with col2:
    st.metric("마지막 업데이트", datetime.now().strftime("%H:%M:%S"))

customers, transactions = load_data()