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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# SHAP import (선택적)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    st.warning("⚠️ SHAP 라이브러리가 설치되지 않았습니다. SHAP 기능을 사용하려면 `pip install shap`을 실행하세요.")

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

# SHAP Explainer 생성 (캐싱)
@st.cache_resource
def get_shap_explainer(_model, _customers_df, _feature_cols):
    """SHAP Explainer 생성 및 캐싱"""
    if not HAS_SHAP:
        return None, None, None
    
    try:
        # 학습 데이터 준비
        X_train = _customers_df[_feature_cols].fillna(0).values
        explainer = shap.TreeExplainer(_model)
        return explainer, X_train, _feature_cols
    except Exception as e:
        st.warning(f"SHAP Explainer 생성 실패: {e}")
        return None, None, None

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
    
    # 글로벌 Feature Importance (SHAP 기반)
    if HAS_SHAP:
        st.subheader("📊 전체 고객 기준 피처 중요도 (SHAP)")
        explainer, X_train, _ = get_shap_explainer(model, customers_df, feature_cols)
        
        if explainer is not None:
            try:
                # 샘플링하여 계산 속도 향상 (전체 데이터가 많을 경우)
                sample_size = min(500, len(customers_df))
                sample_indices = np.random.choice(len(customers_df), sample_size, replace=False)
                X_sample = customers_df[feature_cols].iloc[sample_indices].fillna(0).values
                
                # SHAP values 계산
                shap_values_sample = explainer.shap_values(X_sample)
                
                # 이진 분류인 경우 클래스 1 (해지)의 SHAP 값 사용
                if isinstance(shap_values_sample, list):
                    shap_values_sample = shap_values_sample[1]
                
                # numpy 배열로 변환
                shap_values_sample = np.array(shap_values_sample)
                
                # 2D 배열인지 확인 (샘플 수 × 피처 개수)
                if len(shap_values_sample.shape) == 1:
                    # 1D인 경우 2D로 변환 (1 × 피처 개수)
                    shap_values_sample = shap_values_sample.reshape(1, -1)
                
                # 평균 절댓값 SHAP 계산 (axis=0: 각 피처별로 평균)
                mean_abs_shap = np.abs(shap_values_sample).mean(axis=0)
                
                # 길이 확인 및 조정
                if len(mean_abs_shap) != len(feature_cols):
                    st.warning(f"SHAP values 피처 개수 불일치: {len(mean_abs_shap)} vs {len(feature_cols)}")
                    min_len = min(len(mean_abs_shap), len(feature_cols))
                    mean_abs_shap = mean_abs_shap[:min_len]
                    feature_cols_adjusted = feature_cols[:min_len]
                else:
                    feature_cols_adjusted = feature_cols
                
                feature_importance_global = pd.DataFrame({
                    'feature': feature_cols_adjusted,
                    'importance': mean_abs_shap
                }).sort_values('importance', ascending=True)
                
                # Bar chart
                fig = px.barh(
                    feature_importance_global, 
                    x='importance', 
                    y='feature',
                    title='모델 피처 중요도 (SHAP 기반)',
                    labels={'importance': '평균 영향도', 'feature': '피처'},
                    color='importance',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP 글로벌 분석 실패: {e}")

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
    
    # SHAP 분석 섹션
    if HAS_SHAP:
        st.divider()
        st.subheader("🔍 모델 해석: 왜 이 고객이 위험한가?")
        
        explainer, X_train, _ = get_shap_explainer(model, customers_df, feature_cols)
        
        if explainer is not None:
            try:
                # 선택된 고객 데이터 준비
                selected_customer_data = customers_df[
                    customers_df['customer_id'] == customer_id
                ][feature_cols].fillna(0)
                
                if len(selected_customer_data) == 0:
                    st.warning("고객 데이터를 찾을 수 없습니다.")
                else:
                    # SHAP values 계산
                    shap_values = explainer.shap_values(selected_customer_data.values)
                    
                    # 이진 분류인 경우 클래스 1 (해지)의 SHAP 값 사용
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    
                    # numpy 배열로 변환
                    shap_values = np.array(shap_values)
                    
                    # 2D 배열인 경우 첫 번째 행만 추출 (1D 배열로)
                    if len(shap_values.shape) > 1:
                        shap_values = shap_values[0]
                    
                    # 1D 배열로 확실히 변환하고 길이 확인
                    shap_values = shap_values.flatten()
                    if len(shap_values) != len(feature_cols):
                        st.error(f"SHAP values 길이 불일치: {len(shap_values)} vs {len(feature_cols)}")
                        shap_values = shap_values[:len(feature_cols)] if len(shap_values) > len(feature_cols) else np.pad(shap_values, (0, len(feature_cols) - len(shap_values)))
                    
                    # Expected value 가져오기
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, (list, np.ndarray)):
                        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                    
                    # Feature importance DataFrame 생성 (1D 배열로 확실히 변환)
                    feature_importance = pd.DataFrame({
                        'feature': feature_cols,
                        'shap_value': shap_values
                    })
                    feature_importance['abs_shap'] = feature_importance['shap_value'].abs()
                    feature_importance = feature_importance.sort_values('abs_shap', ascending=False)
                    
                    # 1. 텍스트 기반 해석 (메인)
                    st.markdown("#### 1️⃣ 주요 위험 요인 분석")
                    
                    col1, col2, col3 = st.columns(3)
                    top_features = feature_importance.head(3)
                    
                    feature_names_kr = {
                        'age': '나이',
                        'total_spent': '총 구매액',
                        'total_orders': '총 주문 수',
                        'avg_order_value': '평균 주문액',
                        'last_order_days': '최근 주문일',
                        'support_tickets': '고객센터 문의'
                    }
                    
                    for idx, (_, row) in enumerate(top_features.iterrows()):
                        with [col1, col2, col3][idx]:
                            feature_name = row['feature']
                            feature_name_kr = feature_names_kr.get(feature_name, feature_name)
                            shap_val = row['shap_value']
                            direction = "↑ 증가" if shap_val > 0 else "↓ 감소"
                            
                            current_value = selected_customer_data[feature_name].values[0]
                            
                            st.metric(
                                f"{idx+1}. {feature_name_kr}",
                                f"{current_value:.1f}",
                                delta=f"{abs(shap_val):.3f} ({direction})"
                            )
                    
                    # 해석 텍스트
                    top_feature = feature_importance.iloc[0]
                    top_feature_name = feature_names_kr.get(top_feature['feature'], top_feature['feature'])
                    top_feature_value = selected_customer_data[top_feature['feature']].values[0]
                    
                    second_feature = feature_importance.iloc[1] if len(feature_importance) > 1 else None
                    third_feature = feature_importance.iloc[2] if len(feature_importance) > 2 else None
                    
                    interpretation = f"""
**🎯 이 고객의 해지 위험 원인:**

**상위 위험 요인: {top_feature_name.upper()}**
- 현재값: {top_feature_value:.1f}
- 영향도: {top_feature['shap_value']:.3f} ({'해지 위험 증가' if top_feature['shap_value'] > 0 else '해지 위험 감소'})
"""
                    
                    if second_feature is not None:
                        second_feature_name = feature_names_kr.get(second_feature['feature'], second_feature['feature'])
                        second_feature_value = selected_customer_data[second_feature['feature']].values[0]
                        interpretation += f"""
**보조 요인: {second_feature_name.upper()}**
- 현재값: {second_feature_value:.1f}
- 영향도: {second_feature['shap_value']:.3f} ({'해지 위험 증가' if second_feature['shap_value'] > 0 else '해지 위험 감소'})
"""
                    
                    if third_feature is not None:
                        third_feature_name = feature_names_kr.get(third_feature['feature'], third_feature['feature'])
                        third_feature_value = selected_customer_data[third_feature['feature']].values[0]
                        interpretation += f"""
**추가 요인: {third_feature_name.upper()}**
- 현재값: {third_feature_value:.1f}
- 영향도: {third_feature['shap_value']:.3f} ({'해지 위험 증가' if third_feature['shap_value'] > 0 else '해지 위험 감소'})
"""
                    
                    # 권장 액션
                    if top_feature['shap_value'] > 0:
                        action_suggestion = "💡 **권장 액션:**\n"
                        if top_feature['feature'] == 'support_tickets':
                            action_suggestion += "- 고객센터 문의 문제 해결 우선\n- 고객 만족도 개선 프로그램 제공\n- 할인 쿠폰 또는 특별 프로모션 제공"
                        elif top_feature['feature'] == 'last_order_days':
                            action_suggestion += "- 재참여 유도 메일/알림 발송\n- 신규 프로젝트 추천\n- 맞춤형 프로모션 제공"
                        elif top_feature['feature'] == 'total_spent':
                            action_suggestion += "- 구매 촉진 프로모션 제공\n- 충성 고객 프로그램 안내\n- 맞춤형 서비스 추천"
                        else:
                            action_suggestion += "- 개인 맞춤형 고객 관리 프로그램 참여 권유\n- 고객 만족도 조사 및 피드백 수집"
                        
                        interpretation += f"\n{action_suggestion}"
                    
                    st.info(interpretation)
                    
                    # 2. Waterfall Plot (Plotly 기반)
                    st.markdown("#### 2️⃣ 해지 확률 분해 (Waterfall)")
                    
                    # 기본값에서 시작하여 각 피처의 기여도를 순차적으로 더함
                    base_value = expected_value if isinstance(expected_value, (int, float)) else 0.0
                    
                    # Waterfall chart 데이터 준비
                    waterfall_data = []
                    cumulative = float(base_value) if isinstance(base_value, (int, float)) else 0.0
                    
                    # feature_importance를 shap_value 순으로 정렬 (절댓값 기준)
                    for _, row in feature_importance.iterrows():
                        feature_name_kr = feature_names_kr.get(row['feature'], row['feature'])
                        shap_val = float(row['shap_value'])
                        waterfall_data.append({
                            'feature': feature_name_kr,
                            'shap_value': shap_val,
                            'cumulative': cumulative
                        })
                        cumulative += shap_val
                    
                    # Plotly Waterfall chart
                    base_val = float(base_value) if isinstance(base_value, (int, float)) else 0.0
                    final_value = float(cumulative)
                    
                    fig_waterfall = go.Figure(go.Waterfall(
                        orientation="v",
                        measure=["absolute"] + ["relative"] * len(waterfall_data) + ["total"],
                        x=["기본값"] + [w['feature'] for w in waterfall_data] + ["최종 예측"],
                        textposition="outside",
                        text=[f"{base_val:.2%}"] + 
                             [f"+{w['shap_value']:.2%}" if w['shap_value'] > 0 else f"{w['shap_value']:.2%}" 
                              for w in waterfall_data] + 
                             [f"{final_value:.2%}"],
                        y=[base_val] + [w['shap_value'] for w in waterfall_data] + [final_value],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        increasing={"marker": {"color": "#e74c3c"}},
                        decreasing={"marker": {"color": "#2ecc71"}},
                    ))
                    
                    fig_waterfall.update_layout(
                        title=f"해지 확률 분해 (기본값: {base_val:.2%} → 최종: {final_value:.2%})",
                        showlegend=False,
                        height=500,
                        xaxis_title="피처",
                        yaxis_title="해지 확률"
                    )
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                    
                    # 3. Feature Importance Bar Chart
                    st.markdown("#### 3️⃣ 피처 중요도 (SHAP 기반)")
                    
                    fig_importance = px.bar(
                        feature_importance,
                        x='abs_shap',
                        y='feature',
                        orientation='h',
                        title='피처별 해지 위험 영향도',
                        labels={'abs_shap': '절댓값 SHAP (영향도)', 'feature': '피처'},
                        color='abs_shap',
                        color_continuous_scale='Reds'
                    )
                    fig_importance.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
            except Exception as e:
                st.error(f"SHAP 계산 오류: {e}")
                import traceback
                st.code(traceback.format_exc())

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
