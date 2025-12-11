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
    """샘플 데이터 로드 (캐싱)"""
    customer_df, seller_df, transaction_df = generate_all_sample_data(
        n_customers=1000,
        n_sellers=200,
        n_transactions=5000
    )
    return customer_df, seller_df, transaction_df


def calculate_recent_churn_rate(customer_df, transaction_df, days=7):
    """
    최근 N일 평균 해지율 계산
    
    Args:
        customer_df: 고객 데이터프레임
        transaction_df: 거래 데이터프레임
        days: 최근 며칠간 (기본 7일)
    
    Returns:
        float: 최근 해지율 (%)
    """
    from datetime import datetime, timedelta
    
    # 최근 N일 이내 거래 고객 필터링
    # 입력 데이터프레임을 수정하지 않도록 복사본 생성
    transaction_df_copy = transaction_df.copy()
    cutoff_date = datetime.now() - timedelta(days=days)
    transaction_df_copy['transaction_date'] = pd.to_datetime(transaction_df_copy['transaction_date'])
    recent_customers = transaction_df_copy[
        transaction_df_copy['transaction_date'] >= cutoff_date
    ]['customer_id'].unique()
    
    # 최근 거래 고객 중 해지 고객 비율
    if len(recent_customers) > 0:
        recent_customer_df = customer_df[customer_df['customer_id'].isin(recent_customers)]
        if len(recent_customer_df) > 0 and 'churn' in recent_customer_df.columns:
            return recent_customer_df['churn'].mean() * 100
    
    # 대체: 전체 해지율의 80% (최근 데이터가 적을 경우)
    if 'churn' in customer_df.columns:
        return customer_df['churn'].mean() * 100 * 0.8
    
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
        customer_df, seller_df, transaction_df = load_sample_data()
        predictor = load_predictor()
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 데이터 필터
        st.subheader("📊 필터")
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
        
        st.divider()
        st.markdown("**💡 팁**: 필터를 조정하여 특정 고객 그룹을 분석할 수 있습니다.")
    
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
    
    # 상단 메트릭 요약
    show_metrics_summary(filtered_df, customer_df, transaction_df)
    
    # 탭 네비게이션
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 개별 고객 조회",
        "📊 세그먼트 분석",
        "🧪 A/B 테스트",
        "📦 배치 분석"
    ])
    
    with tab1:
        show_customer_detail(filtered_df, predictor)
    
    with tab2:
        show_segment_analysis(filtered_df, predictor)
    
    with tab3:
        show_ab_test(filtered_df, predictor)
    
    with tab4:
        show_batch_analysis(filtered_df, customer_df, seller_df, transaction_df, predictor)


def show_metrics_summary(df, customer_df, transaction_df):
    """상단 메트릭 요약 섹션"""
    st.markdown("### 📊 주요 지표")
    
    # 메트릭 계산
    avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0
    high_risk_count = (df['risk_score'] >= 70).sum() if 'risk_score' in df.columns else 0
    recent_churn_rate = calculate_recent_churn_rate(customer_df, transaction_df, days=7)
    
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
        segment_col = 'subscription_type'
    elif segment_type == "지역":
        segment_col = 'region'
    else:
        segment_col = 'customer_type'
    
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
        segment_detail = df.groupby(segment_col).agg({
            'risk_score': ['mean', 'std', 'min', 'max'],
            'predicted_churn': 'sum' if 'predicted_churn' in df.columns else 'count',
            'total_orders': 'mean',
            'total_spent': 'mean'
        }).round(2)
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


def show_batch_analysis(df, customer_df, seller_df, transaction_df, predictor):
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
        
        data_type = st.selectbox("데이터 유형", ["고객 데이터", "판매자 데이터", "거래 데이터"])
        
        if data_type == "고객 데이터":
            st.dataframe(customer_df, use_container_width=True)
        elif data_type == "판매자 데이터":
            st.dataframe(seller_df, use_container_width=True)
        else:
            st.dataframe(transaction_df, use_container_width=True)


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




if __name__ == "__main__":
    main()

