"""
시각화 유틸리티 모듈
Plotly를 사용한 대시보드 차트 생성 함수들
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import shap


def create_risk_score_gauge(risk_score):
    """
    리스크 스코어 게이지 차트 생성
    
    Args:
        risk_score: 리스크 스코어 (0-100)
    
    Returns:
        plotly.graph_objects.Figure: 게이지 차트
    """
    # 색상 결정
    if risk_score < 30:
        color = "green"
        label = "낮음"
    elif risk_score < 70:
        color = "orange"
        label = "보통"
    else:
        color = "red"
        label = "높음"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"리스크 스코어 ({label})"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_churn_distribution_chart(df):
    """
    해지 분포 차트 생성
    
    Args:
        df: 고객 데이터프레임 (churn 컬럼 포함)
    
    Returns:
        plotly.graph_objects.Figure: 파이 차트
    """
    churn_counts = df['churn'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['유지', '해지'],
        values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
        hole=0.4,
        marker_colors=['#2ecc71', '#e74c3c']
    )])
    
    fig.update_layout(
        title="고객 해지 분포",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_risk_score_distribution(df):
    """
    리스크 스코어 분포 히스토그램 생성
    
    Args:
        df: 고객 데이터프레임 (risk_score 컬럼 포함)
    
    Returns:
        plotly.graph_objects.Figure: 히스토그램
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['risk_score'],
        nbinsx=30,
        marker_color='#3498db',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="리스크 스코어 분포",
        xaxis_title="리스크 스코어",
        yaxis_title="고객 수",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_feature_importance_chart(feature_importance_dict, top_n=10):
    """
    특성 중요도 차트 생성
    
    Args:
        feature_importance_dict: 특성명과 중요도 딕셔너리
        top_n: 상위 N개 특성만 표시
    
    Returns:
        plotly.graph_objects.Figure: 바 차트
    """
    # 상위 N개 선택
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features) if sorted_features else ([], [])
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker_color='#9b59b6'
        )
    ])
    
    fig.update_layout(
        title="특성 중요도 (상위 10개)",
        xaxis_title="중요도",
        yaxis_title="특성",
        height=400,
        margin=dict(l=150, r=20, t=40, b=20)
    )
    
    return fig


def create_customer_segmentation_chart(df):
    """
    고객 세그먼트별 해지율 차트
    
    Args:
        df: 고객 데이터프레임
    
    Returns:
        plotly.graph_objects.Figure: 바 차트
    """
    if 'subscription_type' in df.columns:
        segment_col = 'subscription_type'
    elif 'customer_type' in df.columns:
        segment_col = 'customer_type'
    else:
        # 기본 차트 반환
        return create_churn_distribution_chart(df)
    
    segment_churn = df.groupby(segment_col)['churn'].agg(['mean', 'count']).reset_index()
    segment_churn['churn_rate'] = (segment_churn['mean'] * 100).round(2)
    
    fig = go.Figure(data=[
        go.Bar(
            x=segment_churn[segment_col],
            y=segment_churn['churn_rate'],
            marker_color='#e67e22',
            text=segment_churn['churn_rate'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"{segment_col}별 해지율",
        xaxis_title=segment_col,
        yaxis_title="해지율 (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_time_series_churn(df, date_col='join_date'):
    """
    시간별 해지 추이 차트
    
    Args:
        df: 고객 데이터프레임
        date_col: 날짜 컬럼명
    
    Returns:
        plotly.graph_objects.Figure: 라인 차트
    """
    if date_col not in df.columns:
        return None
    
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy['year_month'] = df_copy[date_col].dt.to_period('M').astype(str)
    
    monthly_churn = df_copy.groupby('year_month')['churn'].agg(['mean', 'count']).reset_index()
    monthly_churn['churn_rate'] = (monthly_churn['mean'] * 100).round(2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_churn['year_month'],
        y=monthly_churn['churn_rate'],
        mode='lines+markers',
        name='해지율',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="월별 해지율 추이",
        xaxis_title="월",
        yaxis_title="해지율 (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(tickangle=-45)
    )
    
    return fig


def create_shap_summary_plot(model, X, feature_names=None, max_display=10):
    """
    SHAP 요약 플롯 생성
    
    Args:
        model: 학습된 모델
        X: 입력 데이터
        feature_names: 특성명 리스트
        max_display: 최대 표시 특성 수
    
    Returns:
        matplotlib figure 또는 None
    """
    try:
        # SHAP explainer 생성
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # 2D 배열인 경우 (이진 분류)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 클래스 1 (해지)의 SHAP 값 사용
        
        # 요약 플롯 생성
        fig = shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        return fig
    except Exception as e:
        print(f"SHAP 플롯 생성 오류: {e}")
        return None


def create_correlation_heatmap(df, numeric_cols=None):
    """
    상관관계 히트맵 생성
    
    Args:
        df: 데이터프레임
        numeric_cols: 숫자형 컬럼 리스트 (None이면 자동 선택)
    
    Returns:
        plotly.graph_objects.Figure: 히트맵
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 해지 관련 컬럼만 선택
    relevant_cols = [col for col in numeric_cols if col not in ['customer_id']]
    corr_matrix = df[relevant_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="상관계수")
    ))
    
    fig.update_layout(
        title="특성 간 상관관계",
        height=500,
        margin=dict(l=100, r=20, t=40, b=100)
    )
    
    return fig

