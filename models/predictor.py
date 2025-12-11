"""
모델 예측 모듈
사전 학습된 Decision Tree 모델을 로드하고 예측을 수행합니다.
"""
import pickle
import os
import pandas as pd
import numpy as np
from pathlib import Path


class ChurnPredictor:
    """고객 해지 예측 모델 클래스"""
    
    def __init__(self, model_path=None):
        """
        모델 초기화
        
        Args:
            model_path: 모델 파일 경로 (None이면 기본 경로 사용)
        """
        if model_path is None:
            # 프로젝트 루트 기준으로 모델 경로 설정
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / "data" / "model.pkl"
        
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    # 모델 데이터 구조에 따라 분기
                    if isinstance(model_data, dict):
                        self.model = model_data.get('model')
                        self.feature_names = model_data.get('feature_names')
                    else:
                        self.model = model_data
                print(f"모델 로드 완료: {self.model_path}")
            else:
                print(f"경고: 모델 파일을 찾을 수 없습니다: {self.model_path}")
                print("기본 모델을 생성합니다...")
                self._create_default_model()
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("기본 모델을 생성합니다...")
            self._create_default_model()
    
    def _create_default_model(self):
        """기본 모델 생성 (모델 파일이 없을 경우)"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import StandardScaler
        
        # 더미 모델 생성
        self.model = DecisionTreeClassifier(max_depth=5, random_state=42)
        # 더미 데이터로 학습 (실제로는 학습 스크립트에서 생성된 모델 사용)
        X_dummy = np.random.rand(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        self.model.fit(X_dummy, y_dummy)
        
        self.feature_names = [
            'age', 'total_spent', 'total_orders', 'avg_order_value',
            'last_order_days', 'support_tickets', 'subscription_type_encoded',
            'customer_type_encoded', 'region_encoded'
        ]
    
    def preprocess_features(self, df):
        """
        입력 데이터 전처리
        
        Args:
            df: 고객 데이터프레임
        
        Returns:
            numpy array: 전처리된 특성 배열
        """
        # 데이터 복사
        data = df.copy()
        
        # 범주형 변수 인코딩
        if 'subscription_type' in data.columns:
            subscription_map = {'Basic': 0, 'Premium': 1, 'Enterprise': 2}
            data['subscription_type_encoded'] = data['subscription_type'].map(subscription_map).fillna(0)
        
        if 'customer_type' in data.columns:
            data['customer_type_encoded'] = (data['customer_type'] == '기업').astype(int)
        
        if 'region' in data.columns:
            region_map = {'서울': 0, '경기': 1, '부산': 2, '인천': 3, '대구': 4, '기타': 5}
            data['region_encoded'] = data['region'].map(region_map).fillna(5)
        
        # 사용할 특성 선택
        feature_cols = [
            'age', 'total_spent', 'total_orders', 'avg_order_value',
            'last_order_days', 'support_tickets', 'subscription_type_encoded',
            'customer_type_encoded', 'region_encoded'
        ]
        
        # 존재하는 특성만 선택
        available_cols = [col for col in feature_cols if col in data.columns]
        X = data[available_cols].fillna(0).values
        
        return X, available_cols
    
    def predict(self, df):
        """
        해지 예측 수행
        
        Args:
            df: 고객 데이터프레임 (단일 또는 여러 행)
        
        Returns:
            dict: 예측 결과 (churn, probability, risk_score)
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 전처리
        X, feature_names = self.preprocess_features(df)
        
        # 예측
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # 해지 확률 (클래스 1의 확률)
        churn_probs = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        # 리스크 스코어 (0-100)
        risk_scores = (churn_probs * 100).round(2)
        
        results = {
            'churn': predictions,
            'churn_probability': churn_probs,
            'risk_score': risk_scores
        }
        
        return results
    
    def predict_single(self, customer_data):
        """
        단일 고객 예측
        
        Args:
            customer_data: 단일 고객 데이터 (dict 또는 Series)
        
        Returns:
            dict: 예측 결과
        """
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = pd.DataFrame([customer_data])
        
        results = self.predict(df)
        
        return {
            'churn': results['churn'][0],
            'churn_probability': results['churn_probability'][0],
            'risk_score': results['risk_score'][0]
        }
    
    def get_feature_importance(self):
        """
        특성 중요도 반환
        
        Returns:
            dict: 특성명과 중요도
        """
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
            importance_dict = dict(zip(feature_names, self.model.feature_importances_))
            # 중요도 순으로 정렬
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}


if __name__ == "__main__":
    # 테스트
    predictor = ChurnPredictor()
    print("모델 로드 완료")
    
    # 더미 데이터로 테스트
    from data.sample_data import generate_customer_data
    test_data = generate_customer_data(n_customers=5)
    results = predictor.predict(test_data)
    
    print("\n예측 결과:")
    print(results)
    
    print("\n특성 중요도:")
    print(predictor.get_feature_importance())

