"""
모델 학습 스크립트
Decision Tree 모델을 학습하고 저장합니다.
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from data.sample_data import generate_customer_data


def prepare_features(df):
    """
    특성 준비 및 전처리
    
    Args:
        df: 고객 데이터프레임
    
    Returns:
        tuple: (X, y, feature_names)
    """
    data = df.copy()
    
    # 범주형 변수 인코딩
    subscription_map = {'Basic': 0, 'Premium': 1, 'Enterprise': 2}
    data['subscription_type_encoded'] = data['subscription_type'].map(subscription_map).fillna(0)
    
    data['customer_type_encoded'] = (data['customer_type'] == '기업').astype(int)
    
    region_map = {'서울': 0, '경기': 1, '부산': 2, '인천': 3, '대구': 4, '기타': 5}
    data['region_encoded'] = data['region'].map(region_map).fillna(5)
    
    # 특성 선택
    feature_cols = [
        'age', 'total_spent', 'total_orders', 'avg_order_value',
        'last_order_days', 'support_tickets', 'subscription_type_encoded',
        'customer_type_encoded', 'region_encoded'
    ]
    
    X = data[feature_cols].fillna(0).values
    y = data['churn'].values
    
    return X, y, feature_cols


def train_model(n_samples=5000, test_size=0.2, random_state=42):
    """
    모델 학습
    
    Args:
        n_samples: 학습 데이터 샘플 수
        test_size: 테스트 데이터 비율
        random_state: 랜덤 시드
    
    Returns:
        tuple: (model, X_test, y_test, feature_names)
    """
    print("데이터 생성 중...")
    df = generate_customer_data(n_customers=n_samples, seed=random_state)
    
    print("특성 준비 중...")
    X, y, feature_names = prepare_features(df)
    
    print("데이터 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("모델 학습 중...")
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        class_weight='balanced'  # 클래스 불균형 처리
    )
    
    model.fit(X_train, y_train)
    
    # 평가
    print("\n모델 평가:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"정확도: {accuracy:.4f}")
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred))
    print("\n혼동 행렬:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, X_test, y_test, feature_names


def save_model(model, feature_names, model_path=None):
    """
    모델 저장
    
    Args:
        model: 학습된 모델
        feature_names: 특성명 리스트
        model_path: 저장 경로
    """
    if model_path is None:
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / "data" / "model.pkl"
    
    # 모델 디렉토리 생성
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 모델과 특성명을 함께 저장
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n모델 저장 완료: {model_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("Decision Tree 모델 학습 시작")
    print("=" * 50)
    
    # 모델 학습
    model, X_test, y_test, feature_names = train_model(
        n_samples=5000,
        test_size=0.2,
        random_state=42
    )
    
    # 모델 저장
    save_model(model, feature_names)
    
    # 특성 중요도 출력
    print("\n특성 중요도:")
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    print("\n" + "=" * 50)
    print("모델 학습 완료!")
    print("=" * 50)

