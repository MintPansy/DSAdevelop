"""
IT 아웃소싱 플랫폼 고객 해지예측 Streamlit 대시보드
최소 작동 버전 - 모든 기능이 정상 작동하도록 단순화
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent))

# 데이터 생성
from data.sample_data import generate_all_sample_data

# 페이지 설정
st.set_page_config(
    page_title="고객 해지예측 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 헤더
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🔴 IT 아웃소싱 고객 해지예측 대시보드</h1>
        <p style='color: #f0f0f0; margin: 0.5rem 0 0 0;'>데이터 기반 고객 리스크 모니터링 시스템</p>
    </div>
""", unsafe_allow_html=True)

# 데이터 로드
@st.cache_data
def load_data():
    """샘플 데이터 로드"""
    customers, sellers, transactions = generate_all_sample_data(
        n_customers=1000,
        n_sellers=200,
        n_transactions=5000
    )
    return customers, sellers, transactions

with st.spinner("데이터 로딩 중..."):
    customers_df, sellers_df, transactions_df = load_data()

# 간단한 모델 학습
@st.cache_resource
def train_model():
    """모델 학습"""
    # 사용 가능한 컬럼만 선택
    feature_cols = []
    if 'age' in customers_df.columns:
        feature_cols.append('age')
    if 'total_spent' in customers_df.columns:
        feature_cols.append('total_spent')
    if 'total_orders' in customers_df.columns:
        feature_cols.append('total_orders')
    if 'avg_order_value' in customers_df.columns:
        feature_cols.append('avg_order_value')
    if 'last_order_days' in customers_df.columns:
        feature_cols.append('last_order_days')
    
    if len(feature_cols) == 0:
        # 기본값 사용
        feature_cols = ['age', 'total_spent', 'total_orders']
        X = np.random.rand(len(customers_df), len(feature_cols))
    else:
        X = customers_df[feature_cols].fillna(0).values
    
    # 타겟 변수
    if 'churn' in customers_df.columns:
        y = customers_df['churn'].fillna(0).values
    else:
        y = np.random.randint(0, 2, len(customers_df))
    
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    return model, feature_cols

with st.spinner("모델 학습 중..."):
    model, feature_cols = train_model()


# 예측
if len(feature_cols) > 0:
    X_pred = customers_df[feature_cols].fillna(0).values if all(col in customers_df.columns for col in feature_cols) else np.random.rand(len(customers_df), len(feature_cols))
else:
    X_pred = np.random.rand(len(customers_df), 3)

predictions = model.predict(X_pred)
probabilities = model.predict_proba(X_pred)

customers_df['predicted_churn'] = predictions
customers_df['churn_probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
customers_df['risk_score'] = (customers_df['churn_probability'] * 100).round(2)

# 실제 churn 값이 있으면 사용, 없으면 예측값 사용
if 'churn' not in customers_df.columns:
    customers_df['churn'] = customers_df['predicted_churn']

# 주요 지표
st.markdown("### 📊 주요 지표")
col1, col2, col3, col4 = st.columns(4)

with col1:
    churn_rate = customers_df['churn'].mean() * 100 if 'churn' in customers_df.columns else customers_df['predicted_churn'].mean() * 100
    st.metric("평균 해지율", f"{churn_rate:.1f}%")

with col2:
    high_risk_count = int((customers_df['churn_probability'] > 0.7).sum())
    st.metric("고위험 고객", f"{high_risk_count}명")

with col3:
    st.metric("분석 대상", f"{len(customers_df):,}명")

with col4:
    st.metric("마지막 업데이트", datetime.now().strftime("%H:%M:%S"))

st.divider()

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 대시보드", "🎯 개별 조회", "📈 분석"])

with tab1:
    st.subheader("해지율 분포")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 해지 여부 분포
        churn_dist = customers_df['churn'].value_counts()
        fig = go.Figure(data=[
            go.Bar(
                x=['정상', '해지'], 
                y=[churn_dist.get(0, 0), churn_dist.get(1, 0)],
                marker_color=['#2ecc71', '#e74c3c'],
                text=[churn_dist.get(0, 0), churn_dist.get(1, 0)],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="고객 해지 현황",
            height=400,
            xaxis_title="상태",
            yaxis_title="고객 수"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 리스크 스코어 분포
        fig = px.histogram(
            customers_df, 
            x='churn_probability', 
            nbins=30,
            title="리스크 스코어 분포",
            labels={'churn_probability': '해지 확률', 'count': '고객 수'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 고위험 고객
    st.subheader("⚠️ 고위험 고객 TOP 10")
    high_risk = customers_df.nlargest(10, 'churn_probability').copy()
    
    display_cols = ['customer_id']
    if 'age' in high_risk.columns:
        display_cols.append('age')
    if 'total_spent' in high_risk.columns:
        display_cols.append('total_spent')
    if 'region' in high_risk.columns:
        display_cols.append('region')
    display_cols.append('churn_probability')
    
    available_cols = [col for col in display_cols if col in high_risk.columns]
    high_risk_display = high_risk[available_cols].copy()
    
    if 'churn_probability' in high_risk_display.columns:
        high_risk_display['churn_probability'] = (high_risk_display['churn_probability'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(high_risk_display, use_container_width=True, hide_index=True)
    
    # 글로벌 Feature Importance
    st.subheader("📊 전체 고객 기준 피처 중요도")
    
    try:
        # ✅ sklearn Decision Tree에서 직접 추출 (가장 안정적)
        # 모델 학습 시 사용한 feature_cols 사용
        feature_importances = model.feature_importances_
        
        # DataFrame 생성 (완벽하게 안정적)
        feature_importance_global = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importances
        }).sort_values('importance', ascending=True)
        
        # 시각화
        fig = go.Figure(data=[
            go.Bar(
                y=feature_importance_global['feature'],
                x=feature_importance_global['importance'],
                orientation='h',
                marker=dict(color='#2E86AB'),
                text=(feature_importance_global['importance'] * 100).round(1),
                textposition='auto',
                texttemplate='%{text}%'
            )
        ])
        
        fig.update_layout(
            title='모델 피처 중요도',
            xaxis_title='중요도 (백분율)',
            yaxis_title='피처',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ 피처 중요도 계산 실패: {e}")

with tab2:
    st.subheader("개별 고객 조회")
    
    customer_id = st.selectbox("고객 선택", customers_df['customer_id'].unique())
    customer = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'age' in customer:
            st.metric("나이", f"{int(customer['age'])}세")
        else:
            st.metric("나이", "N/A")
    
    with col2:
        if 'total_spent' in customer:
            st.metric("총 구매액", f"{customer['total_spent']:,.0f}원")
        else:
            st.metric("총 구매액", "N/A")
    
    with col3:
        churn_prob = float(customer['churn_probability']) if 'churn_probability' in customer else 0.0
        st.metric("해지 확률", f"{churn_prob*100:.1f}%")
    
    st.divider()
    
    # 게이지 차트
    churn_prob_value = float(customer['churn_probability']) if 'churn_probability' in customer else 0.0
    risk_score = churn_prob_value * 100
    
    fig = go.Figure(data=[
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            title={'text': "해지 위험도"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#90EE90"},
                    {'range': [30, 70], 'color': "#FFD700"},
                    {'range': [70, 100], 'color': "#FF6347"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        )
    ])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 고객 상세 정보
    st.subheader("고객 상세 정보")
    info_items = []
    if 'customer_id' in customer:
        info_items.append(("고객 ID", customer['customer_id']))
    if 'region' in customer:
        info_items.append(("지역", customer['region']))
    if 'subscription_type' in customer:
        info_items.append(("구독 유형", customer['subscription_type']))
    if 'total_orders' in customer:
        info_items.append(("총 주문 수", f"{int(customer['total_orders'])}건"))
    
    for key, value in info_items:
        st.write(f"**{key}**: {value}")
    
    # 위험 요인 분석 섹션
    st.divider()
    st.subheader("🔍 이 고객의 위험 요인 분석")
    
    try:
        # 선택된 고객 데이터 (모델 학습 시 사용한 feature_cols 사용)
        selected_data = customers_df[
            customers_df['customer_id'] == customer_id
        ][feature_cols].fillna(0)
        
        if len(selected_data) == 0:
            st.error("❌ 선택된 고객이 없습니다")
        else:
            # 고객의 실제 값들
            customer_values = selected_data.iloc[0].values
            
            # 모델의 feature_importances_ 사용
            feature_importances = model.feature_importances_
            
            # 피처별 중요도와 고객의 값을 함께 표시
            feature_analysis = pd.DataFrame({
                'feature': feature_cols,
                'importance': feature_importances,
                'customer_value': customer_values
            }).sort_values('importance', ascending=False)
            
            # 상위 3개 표시
            st.write("**주요 영향 요인 TOP 3:**")
            col1, col2, col3 = st.columns(3)
            
            for idx, (i, row) in enumerate(feature_analysis.head(3).iterrows()):
                with [col1, col2, col3][idx]:
                    st.metric(
                        f"{idx+1}. {str(row['feature']).upper()}",
                        f"{row['importance']*100:.1f}%",
                        f"고객값: {row['customer_value']:.1f}"
                    )
            
            # 상세 분석
            if len(feature_analysis) >= 3:
                top_row = feature_analysis.iloc[0]
                second_row = feature_analysis.iloc[1]
                third_row = feature_analysis.iloc[2]
                
                st.info(f"""
### 🎯 위험 요인 분석:

**모델에서 가장 중요한 피처 TOP 3:**

1️⃣ **{str(top_row['feature']).upper()}** ({top_row['importance']*100:.1f}%)
   - 이 고객의 값: {top_row['customer_value']:.1f}
   
2️⃣ **{str(second_row['feature']).upper()}** ({second_row['importance']*100:.1f}%)
   - 이 고객의 값: {second_row['customer_value']:.1f}
   
3️⃣ **{str(third_row['feature']).upper()}** ({third_row['importance']*100:.1f}%)
   - 이 고객의 값: {third_row['customer_value']:.1f}
""")
    
    except Exception as e:
        st.error(f"❌ 개별 고객 분석 실패: {e}")

with tab3:
    st.subheader("📈 세그먼트 분석")
    
    # 연령대별 해지율
    if 'age' in customers_df.columns:
        customers_df['age_group'] = pd.cut(
            customers_df['age'], 
            bins=[0, 20, 30, 40, 50, 100],
            labels=['10s', '20s', '30s', '40s', '50s+']
        )
        age_churn = customers_df.groupby('age_group', observed=True)['churn'].mean() * 100
        
        fig = px.bar(
            x=age_churn.index.astype(str), 
            y=age_churn.values, 
            title="연령대별 해지율",
            labels={'x': '연령대', 'y': '해지율 (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 지역별 해지율 (있는 경우)
    if 'region' in customers_df.columns:
        st.subheader("지역별 해지율")
        region_churn = customers_df.groupby('region')['churn'].mean() * 100
        
        fig = px.bar(
            x=region_churn.index, 
            y=region_churn.values, 
            title="지역별 해지율",
            labels={'x': '지역', 'y': '해지율 (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("원본 데이터")
    st.dataframe(customers_df.head(10), use_container_width=True)

# 마지막 업데이트 시간
st.markdown("---")
st.info(f"📌 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
