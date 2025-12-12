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
def get_shap_explainer():
    """SHAP Explainer 생성"""
    if not HAS_SHAP:
        return None
    
    try:
        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception as e:
        st.error(f"❌ SHAP Explainer 생성 실패: {e}")
        return None

# 안전한 SHAP values 추출 함수 (핵심!)
def safe_extract_shap(shap_values_raw, sample_idx=0):
    """
    SHAP values를 안전하게 1D 배열로 변환
    
    입력:
    - 리스트: [negative_class, positive_class]
      각각 shape: (샘플 수, 피처 수)
    - numpy array: (샘플 수, 피처 수) 또는 (피처 수,)
    
    출력:
    - positive class SHAP values (1D 배열) shape: (피처 수,)
    """
    # Step 1: 리스트 → positive class 선택
    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[1]  # positive class (해지)
    else:
        shap_vals = shap_values_raw
    
    # Step 2: numpy 배열로 변환
    shap_vals = np.asarray(shap_vals)
    
    # Step 3: 첫 번째 샘플 선택 (2D인 경우)
    if len(shap_vals.shape) > 1:
        shap_vals = shap_vals[sample_idx]  # shape: (피처 수,)
    
    # Step 4: ✅ 무조건 1D로 변환 (핵심!)
    shap_vals = shap_vals.flatten()  # shape: (피처 수,)
    
    return shap_vals

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
        explainer = get_shap_explainer()
        
        if explainer is not None:
            try:
                # ✅ Step 1: 데이터 준비 - 모델 학습 시 사용한 피처들만 선택
                # feature_cols는 train_model()에서 반환된 것 사용
                
                # 실제 데이터에서 추출
                sample_size = min(50, len(customers_df))
                X_all = customers_df[feature_cols].head(sample_size).fillna(0)
                
                # ✅ Step 2: 데이터 검증
                if len(X_all) == 0:
                    st.error("❌ 선택된 데이터가 없습니다")
                elif X_all.shape[1] != len(feature_cols):
                    st.error(f"❌ 피처 개수 불일치: X_all.shape[1]={X_all.shape[1]} vs feature_cols={len(feature_cols)}")
                    st.write(f"**디버깅 정보**: X_all.shape={X_all.shape}, feature_cols={feature_cols}")
                else:
                    # ✅ Step 3: SHAP values 계산
                    shap_values_raw = explainer.shap_values(X_all.values)
                    
                    # ✅ Step 4: positive class 추출
                    if isinstance(shap_values_raw, list):
                        shap_vals_all = np.array(shap_values_raw[1])  # (50, 5)
                    else:
                        shap_vals_all = np.array(shap_values_raw)
                    
                    # ✅ Step 5: 길이 검증 (핵심!)
                    if shap_vals_all.shape[1] != len(feature_cols):
                        st.error(f"""
                        ❌ SHAP 피처 개수 불일치!
                        - feature_cols: {len(feature_cols)}개
                        - SHAP values shape[1]: {shap_vals_all.shape[1]}개
                        
                        💡 해결: feature_cols 정의를 확인하세요
                        """)
                        st.write(f"**디버깅 정보**: shap_vals_all.shape={shap_vals_all.shape}, feature_cols={feature_cols}")
                    else:
                        # ✅ Step 6: 평균 계산
                        mean_abs_shap = np.abs(shap_vals_all).mean(axis=0)  # (5,)
                        mean_abs_shap = np.asarray(mean_abs_shap).flatten()
                        
                        # ✅ Step 7: 최종 길이 검증 및 조정
                        min_len = min(len(feature_cols), len(mean_abs_shap))
                        if min_len == 0:
                            st.error("❌ 배열 길이가 0입니다. 데이터를 확인하세요.")
                        else:
                            # 길이가 다르면 조정
                            if len(feature_cols) != len(mean_abs_shap):
                                st.warning(f"⚠️ 배열 길이 불일치 감지: feature_cols={len(feature_cols)}, mean_abs_shap={len(mean_abs_shap)}. 최소 길이({min_len})만큼만 사용합니다.")
                                feature_cols_adjusted = feature_cols[:min_len]
                                mean_abs_shap_adjusted = mean_abs_shap[:min_len]
                            else:
                                feature_cols_adjusted = feature_cols
                                mean_abs_shap_adjusted = mean_abs_shap
                            
                            # ✅ Step 8: DataFrame 생성 (안전하게)
                            feature_importance_global = pd.DataFrame({
                                'feature': list(feature_cols_adjusted),
                                'importance': mean_abs_shap_adjusted
                            }, dtype=object).sort_values('importance', ascending=True)
                        
                            # ✅ Step 9: 시각화
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
                st.error(f"❌ SHAP 글로벌 분석 실패: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.write(f"**디버깅 정보**: {str(e)}")
                st.info("💡 팁: 더미 데이터에서 SHAP 계산이 불안정할 수 있습니다.")
        else:
            st.info("SHAP 분석을 사용할 수 없습니다.")

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
        
        explainer = get_shap_explainer()
        
        if explainer is not None:
            try:
                # ✅ feature_cols 정의 (대시보드와 동일 - 모델 학습 시 사용한 피처)
                # feature_cols는 train_model()에서 반환된 것 사용
                
                # 고객 데이터 선택
                selected_data = customers_df[
                    customers_df['customer_id'] == customer_id
                ][feature_cols].fillna(0)
                
                if len(selected_data) == 0:
                    st.error("❌ 선택된 고객이 없습니다")
                else:
                    # SHAP values 계산
                    shap_values_raw = explainer.shap_values(selected_data.values)
                    
                    # positive class 추출
                    if isinstance(shap_values_raw, list):
                        shap_vals = np.array(shap_values_raw[1])  # (1, 5)
                    else:
                        shap_vals = np.array(shap_values_raw)
                    
                    # ✅ 1D로 변환
                    shap_values_1d = np.asarray(shap_vals).flatten()  # (5,)
                    
                    # ✅ 길이 검증 및 조정
                    min_len = min(len(feature_cols), len(shap_values_1d))
                    if min_len == 0:
                        st.error("❌ 배열 길이가 0입니다. 데이터를 확인하세요.")
                    else:
                        # 길이가 다르면 조정
                        if len(shap_values_1d) != len(feature_cols):
                            st.warning(f"""
                            ⚠️ 배열 길이 불일치 감지!
                            - feature_cols: {len(feature_cols)}개
                            - shap_values: {len(shap_values_1d)}개
                            - 최소 길이({min_len})만큼만 사용합니다.
                            """)
                            st.write(f"**디버깅 정보**: shap_values_1d.shape={shap_values_1d.shape}, feature_cols={feature_cols}")
                            feature_cols_adjusted = feature_cols[:min_len]
                            shap_values_1d_adjusted = shap_values_1d[:min_len]
                        else:
                            feature_cols_adjusted = feature_cols
                            shap_values_1d_adjusted = shap_values_1d
                        
                        # DataFrame 생성 (안전하게)
                        feature_importance = pd.DataFrame({
                            'feature': list(feature_cols_adjusted),
                            'shap_value': shap_values_1d_adjusted,
                            'abs_shap': np.abs(shap_values_1d_adjusted)
                        }, dtype=object).sort_values('abs_shap', ascending=False)
                        
                        # Expected value 가져오기
                        expected_value = explainer.expected_value
                        if isinstance(expected_value, (list, np.ndarray)):
                            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                        expected_value = float(expected_value) if isinstance(expected_value, (int, float, np.number)) else 0.0
                        
                        # 상위 3개 표시
                        st.markdown("#### 1️⃣ 주요 위험 요인 분석")
                        st.write("**주요 위험 요인 TOP 3:**")
                        col1, col2, col3 = st.columns(3)
                        
                        for idx, (i, row) in enumerate(feature_importance.head(3).iterrows()):
                            direction = "📈 증가" if row['shap_value'] > 0 else "📉 감소"
                            with [col1, col2, col3][idx]:
                                st.metric(
                                    f"{idx+1}. {row['feature']}",
                                    f"{row['abs_shap']:.4f}",
                                    delta=direction
                                )
                        
                        # 상세 분석
                        top_feature = feature_importance.iloc[0]
                        second_feature = feature_importance.iloc[1] if len(feature_importance) > 1 else None
                        third_feature = feature_importance.iloc[2] if len(feature_importance) > 2 else None
                        
                        interpretation = f"""
### 🎯 이 고객의 위험 요인 분석:

**1순위: {top_feature['feature'].upper()}**
- 영향도: {top_feature['abs_shap']:.4f}
- 방향: {"증가 ↑" if top_feature['shap_value'] > 0 else "감소 ↓"}
"""
                        
                        if second_feature is not None:
                            interpretation += f"""
**2순위: {second_feature['feature'].upper()}**
- 영향도: {second_feature['abs_shap']:.4f}
- 방향: {"증가 ↑" if second_feature['shap_value'] > 0 else "감소 ↓"}
"""
                        
                        if third_feature is not None:
                            interpretation += f"""
**3순위: {third_feature['feature'].upper()}**
- 영향도: {third_feature['abs_shap']:.4f}
- 방향: {"증가 ↑" if third_feature['shap_value'] > 0 else "감소 ↓"}
"""
                        
                        st.info(interpretation)
                        
                        # 2. Waterfall Plot (Plotly 기반)
                        st.markdown("#### 2️⃣ 해지 확률 분해 (Waterfall)")
                        
                        # 기본값에서 시작하여 각 피처의 기여도를 순차적으로 더함
                        base_val = expected_value
                        
                        # Waterfall chart 데이터 준비
                        waterfall_data = []
                        cumulative = base_val
                        
                        # feature_importance를 shap_value 순으로 정렬 (절댓값 기준)
                        for _, row in feature_importance.iterrows():
                            shap_val = float(row['shap_value'])
                            waterfall_data.append({
                                'feature': str(row['feature']),  # 문자열로 변환
                                'shap_value': shap_val,
                                'cumulative': cumulative
                            })
                            cumulative += shap_val
                        
                        # Plotly Waterfall chart
                        final_value = cumulative
                        
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
                st.error(f"❌ 개별 고객 분석 실패: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.write(f"**디버깅 정보**: {str(e)}")
                st.info("💡 팁: 선택한 고객 데이터를 확인해주세요.")
        else:
            st.warning("SHAP 분석을 사용할 수 없습니다.")

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
