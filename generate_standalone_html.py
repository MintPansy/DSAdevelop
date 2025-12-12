"""
Streamlit 대시보드의 Plotly 차트들을 독립적인 HTML 파일로 변환하는 스크립트
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# 현재 스크립트의 디렉토리를 작업 디렉토리로 설정
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(script_dir))

from data.sample_data import generate_all_sample_data
from models.predictor import ChurnPredictor
from utils.visualization import (
    create_churn_distribution_chart,
    create_risk_score_distribution,
    create_feature_importance_chart,
    create_customer_segmentation_chart,
    create_correlation_heatmap
)


def generate_standalone_html():
    """모든 시각화를 포함한 독립적인 HTML 파일 생성"""
    
    print("데이터 생성 중...")
    # 샘플 데이터 생성
    customer_df, seller_df, transaction_df = generate_all_sample_data(
        n_customers=1000,
        n_sellers=200,
        n_transactions=5000
    )
    
    print("모델 로드 중...")
    # 모델 로드 및 예측
    predictor = ChurnPredictor()
    predictions = predictor.predict(customer_df)
    
    # 예측 결과를 데이터프레임에 추가
    customer_df['risk_score'] = predictions['risk_score']
    customer_df['predicted_churn'] = predictions['churn']
    customer_df['churn_probability'] = predictions['churn_probability']
    
    print("차트 생성 중...")
    
    # 1. 해지 분포도 (bar chart - pie chart)
    chart_df = customer_df.copy()
    chart_df['churn'] = chart_df['predicted_churn']
    fig_churn_dist = create_churn_distribution_chart(chart_df)
    
    # 2. 리스크 스코어 분포 (histogram)
    fig_risk_dist = create_risk_score_distribution(customer_df)
    
    # 3. 특성 중요도 (horizontal bar)
    feature_importance = predictor.get_feature_importance()
    fig_feature_imp = create_feature_importance_chart(feature_importance)
    
    # 4. 상관관계 히트맵 (heatmap)
    fig_correlation = create_correlation_heatmap(customer_df)
    
    # 5. 세그먼트별 해지율 (group bar)
    fig_segment = create_customer_segmentation_chart(customer_df)
    
    # 6. 고위험 고객 리스트 (table) - Plotly Table 생성
    high_risk_customers = customer_df[customer_df['risk_score'] >= 70].sort_values(
        'risk_score', ascending=False
    ).head(50)
    
    display_cols = ['customer_id', 'region', 'subscription_type', 'total_orders', 
                   'last_order_days', 'risk_score', 'churn_probability']
    available_cols = [col for col in display_cols if col in high_risk_customers.columns]
    
    # Plotly Table 생성
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=[col.replace('_', ' ').title() for col in available_cols],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[high_risk_customers[col].tolist() for col in available_cols],
            fill_color='lavender',
            align='left',
            font=dict(size=11),
            format=[None if col in ['customer_id', 'region', 'subscription_type'] 
                   else '.2f' if col in ['risk_score', 'churn_probability'] 
                   else None for col in available_cols]
        )
    )])
    
    fig_table.update_layout(
        title="고위험 고객 리스트 (상위 50명)",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    print("HTML 파일 생성 중...")
    
    # 각 차트를 HTML 문자열로 변환
    html_charts = []
    html_charts.append('<div class="chart-container">')
    html_charts.append('<h2>1. 해지 분포도</h2>')
    html_charts.append(fig_churn_dist.to_html(include_plotlyjs=False, div_id="chart1"))
    html_charts.append('</div>')
    
    html_charts.append('<div class="chart-container">')
    html_charts.append('<h2>2. 리스크 스코어 분포</h2>')
    html_charts.append(fig_risk_dist.to_html(include_plotlyjs=False, div_id="chart2"))
    html_charts.append('</div>')
    
    html_charts.append('<div class="chart-container">')
    html_charts.append('<h2>3. 특성 중요도</h2>')
    html_charts.append(fig_feature_imp.to_html(include_plotlyjs=False, div_id="chart3"))
    html_charts.append('</div>')
    
    html_charts.append('<div class="chart-container">')
    html_charts.append('<h2>4. 상관관계 히트맵</h2>')
    html_charts.append(fig_correlation.to_html(include_plotlyjs=False, div_id="chart4"))
    html_charts.append('</div>')
    
    html_charts.append('<div class="chart-container">')
    html_charts.append('<h2>5. 세그먼트별 해지율</h2>')
    html_charts.append(fig_segment.to_html(include_plotlyjs=False, div_id="chart5"))
    html_charts.append('</div>')
    
    html_charts.append('<div class="chart-container">')
    html_charts.append('<h2>6. 고위험 고객 리스트</h2>')
    html_charts.append(fig_table.to_html(include_plotlyjs=False, div_id="chart6"))
    html_charts.append('</div>')
    
    # 전체 HTML 구조 생성
    full_html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IT 아웃소싱 플랫폼 고객 해지예측 대시보드</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                        sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 30px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        .header p {{
            font-size: 1.2rem;
            color: #f0f0f0;
            font-weight: 300;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .stat-card h3 {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: #333;
        }}
        
        .chart-container {{
            margin-bottom: 50px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        
        .chart-container h2 {{
            font-size: 1.5rem;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .chart-container .js-plotly-plot {{
            width: 100%;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
            margin-top: 40px;
            border-top: 1px solid #ddd;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8rem;
            }}
            
            .summary-stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔴 IT 아웃소싱 고객 해지예측 대시보드</h1>
            <p>데이터 기반 고객 리스크 모니터링 시스템</p>
        </div>
        
        <div class="summary-stats">
            <div class="stat-card">
                <h3>전체 고객 수</h3>
                <div class="value">{len(customer_df):,}명</div>
            </div>
            <div class="stat-card">
                <h3>평균 해지 위험도</h3>
                <div class="value">{customer_df['risk_score'].mean():.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>고위험 고객 수</h3>
                <div class="value">{(customer_df['risk_score'] >= 70).sum():,}명</div>
            </div>
            <div class="stat-card">
                <h3>예상 해지율</h3>
                <div class="value">{customer_df['predicted_churn'].mean() * 100:.2f}%</div>
            </div>
        </div>
        
        {''.join(html_charts)}
        
        <div class="footer">
            <p>생성 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>이 대시보드는 Streamlit 앱에서 추출한 독립적인 HTML 파일입니다.</p>
        </div>
    </div>
</body>
</html>
    """
    
    # HTML 파일 저장
    output_path = Path(__file__).parent.absolute() / "standalone_dashboard.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"\n✅ 독립적인 HTML 파일이 생성되었습니다: {output_path}")
    print(f"   파일 크기: {output_path.stat().st_size / 1024:.2f} KB")
    return output_path


if __name__ == "__main__":
    try:
        output_path = generate_standalone_html()
        print(f"\n🎉 완료! 브라우저에서 {output_path} 파일을 열어 확인하세요.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
