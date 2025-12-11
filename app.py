"""
IT 아웃소싱 플랫폼 고객 해지예측 Streamlit 대시보드
Decision Tree 모델을 활용한 실시간 리스크 스코어 표시
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent))

from data.sample_data import generate_all_sample_data
from models.predictor import ChurnPredictor
from utils.visualization import (
    create_risk_score_gauge,
    create_churn_distribution_chart,
    create_risk_score_distribution,
    create_feature_importance_chart,
    create_customer_segmentation_chart,
    create_time_series_churn,
    create_correlation_heatmap
)

# 페이지 설정
st.set_page_config(
    page_title="고객 해지예측 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """샘플 데이터 로드 (캐싱)"""
    customer_df, seller_df, transaction_df = generate_all_sample_data(
        n_customers=1000,
        n_sellers=200,
        n_transactions=5000
    )
    return customer_df, seller_df, transaction_df


@st.cache_resource
def load_predictor():
    """모델 로드 (캐싱)"""
    return ChurnPredictor()


def main():
    """메인 함수"""
    # 헤더
    st.markdown('<div class="main-header">📊 IT 아웃소싱 플랫폼 고객 해지예측 대시보드</div>', unsafe_allow_html=True)
    
    # 데이터 로드
    with st.spinner("데이터 로딩 중..."):
        customer_df, seller_df, transaction_df = load_sample_data()
        predictor = load_predictor()
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 탭 선택
        page = st.radio(
            "페이지 선택",
            ["대시보드", "개별 고객 조회", "데이터 분석"]
        )
        
        # 데이터 필터
        st.subheader("필터")
        selected_regions = st.multiselect(
            "지역 선택",
            options=customer_df['region'].unique(),
            default=customer_df['region'].unique()
        )
        
        selected_subscription = st.multiselect(
            "구독 유형",
            options=customer_df['subscription_type'].unique(),
            default=customer_df['subscription_type'].unique()
        )
        
        risk_threshold = st.slider(
            "리스크 스코어 임계값",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )
    
    # 필터 적용
    filtered_df = customer_df[
        (customer_df['region'].isin(selected_regions)) &
        (customer_df['subscription_type'].isin(selected_subscription))
    ].copy()
    
    # 예측 수행
    if len(filtered_df) > 0:
        predictions = predictor.predict(filtered_df)
        filtered_df['risk_score'] = predictions['risk_score']
        filtered_df['churn_probability'] = predictions['churn_probability']
        filtered_df['predicted_churn'] = predictions['churn']
    
    # 페이지별 콘텐츠
    if page == "대시보드":
        show_dashboard(filtered_df, predictor)
    elif page == "개별 고객 조회":
        show_customer_detail(filtered_df, predictor)
    elif page == "데이터 분석":
        show_data_analysis(filtered_df, customer_df, seller_df, transaction_df)


def show_dashboard(df, predictor):
    """대시보드 페이지"""
    st.header("📈 전체 현황")
    
    # 주요 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("전체 고객 수", f"{len(df):,}명")
    
    with col2:
        churn_rate = df['predicted_churn'].mean() * 100 if 'predicted_churn' in df.columns else 0
        st.metric("예상 해지율", f"{churn_rate:.2f}%")
    
    with col3:
        high_risk = (df['risk_score'] >= 70).sum() if 'risk_score' in df.columns else 0
        st.metric("고위험 고객", f"{high_risk:,}명")
    
    with col4:
        avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0
        st.metric("평균 리스크 스코어", f"{avg_risk:.2f}")
    
    st.divider()
    
    # 차트 섹션
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("해지 분포")
        if 'predicted_churn' in df.columns:
            chart_df = df.copy()
            chart_df['churn'] = chart_df['predicted_churn']
            fig = create_churn_distribution_chart(chart_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("예측 데이터가 없습니다.")
    
    with col2:
        st.subheader("리스크 스코어 분포")
        if 'risk_score' in df.columns:
            fig = create_risk_score_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("예측 데이터가 없습니다.")
    
    # 고위험 고객 리스트
    st.subheader("⚠️ 고위험 고객 리스트")
    if 'risk_score' in df.columns:
        high_risk_customers = df[df['risk_score'] >= 70].sort_values('risk_score', ascending=False)
        
        if len(high_risk_customers) > 0:
            display_cols = ['customer_id', 'region', 'subscription_type', 'total_orders', 
                          'last_order_days', 'risk_score', 'churn_probability']
            available_cols = [col for col in display_cols if col in high_risk_customers.columns]
            
            st.dataframe(
                high_risk_customers[available_cols].head(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("고위험 고객이 없습니다.")
    else:
        st.info("예측 데이터가 없습니다.")
    
    # 특성 중요도
    st.subheader("특성 중요도")
    feature_importance = predictor.get_feature_importance()
    if feature_importance:
        fig = create_feature_importance_chart(feature_importance)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("특성 중요도 정보가 없습니다.")


def show_customer_detail(df, predictor):
    """개별 고객 조회 페이지"""
    st.header("🔍 개별 고객 조회")
    
    # 고객 선택
    customer_ids = df['customer_id'].tolist()
    selected_id = st.selectbox("고객 ID 선택", customer_ids)
    
    if selected_id:
        customer = df[df['customer_id'] == selected_id].iloc[0]
        
        # 고객 정보 표시
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("고객 정보")
            info_data = {
                "고객 ID": customer['customer_id'],
                "지역": customer['region'],
                "고객 유형": customer['customer_type'],
                "구독 유형": customer['subscription_type'],
                "나이": int(customer['age']),
                "총 주문 수": int(customer['total_orders']),
                "총 구매액": f"{customer['total_spent']:,.0f}원",
                "평균 주문액": f"{customer['avg_order_value']:,.0f}원",
                "마지막 주문일": f"{int(customer['last_order_days'])}일 전",
                "고객센터 문의": int(customer['support_tickets']),
            }
            
            for key, value in info_data.items():
                st.write(f"**{key}**: {value}")
        
        with col2:
            st.subheader("리스크 분석")
            if 'risk_score' in customer:
                risk_score = customer['risk_score']
                fig = create_risk_score_gauge(risk_score)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("해지 확률", f"{customer['churn_probability']*100:.2f}%")
                st.metric("예상 해지 여부", "해지 예상" if customer['predicted_churn'] == 1 else "유지 예상")
            else:
                # 실시간 예측
                result = predictor.predict_single(customer)
                fig = create_risk_score_gauge(result['risk_score'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("해지 확률", f"{result['churn_probability']*100:.2f}%")
                st.metric("예상 해지 여부", "해지 예상" if result['churn'] == 1 else "유지 예상")


def show_data_analysis(df, customer_df, seller_df, transaction_df):
    """데이터 분석 페이지"""
    st.header("📊 데이터 분석")
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["고객 세그먼트", "시간별 추이", "상관관계", "원본 데이터"])
    
    with tab1:
        st.subheader("세그먼트별 해지율")
        fig = create_customer_segmentation_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("월별 해지율 추이")
        fig = create_time_series_churn(customer_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("날짜 데이터가 없습니다.")
    
    with tab3:
        st.subheader("특성 간 상관관계")
        fig = create_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("고객 데이터")
        st.dataframe(customer_df.head(100), use_container_width=True)
        
        st.subheader("판매자 데이터")
        st.dataframe(seller_df.head(100), use_container_width=True)
        
        st.subheader("거래 데이터")
        st.dataframe(transaction_df.head(100), use_container_width=True)


if __name__ == "__main__":
    main()

