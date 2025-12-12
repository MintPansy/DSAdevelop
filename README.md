# 🔴 IT 아웃소싱 고객 해지예측 대시보드

**Live Demo:** [https://dsadevelop-3mgojvzdwykaat2capdvqg.streamlit.app/](배포된 URL)

## 📖 프로젝트 개요

기계학습을 활용한 고객 해지 예측 Streamlit 대시보드입니다.
팀 프로젝트에서 분석한 IT 아웃소싱 거래 데이터 34만 건의 인사이트를 바탕으로,
고객의 해지 위험도를 실시간으로 예측하고 시각화합니다.

### 주요 특징
- **실시간 리스크 예측**: Decision Tree 모델을 활용한 해지 확률 계산
- **인터랙티브 대시보드**: Plotly 기반의 반응형 차트
- **개별 고객 분석**: 고객별 상세 정보 조회 및 위험 요인 파악
- **세그먼트 분석**: 연령대/지역별 해지율 비교 분석
- **즉시 배포**: Streamlit Cloud로 누구나 접근 가능

## 🛠️ 기술 스택

| 분야 | 기술 |
|------|------|
| **웹 프레임워크** | Streamlit 1.28.0 |
| **데이터 처리** | Pandas 2.1.0 |
| **머신러닝** | Scikit-learn 1.3.0 |
| **시각화** | Plotly 5.14.0 |
| **배포** | Streamlit Cloud |

## 📊 데이터 구조

### 고객 데이터 (Customer Data)
- **크기**: 10,000명 (팀 프로젝트 기반 synthetic data)
- **주요 피처**: 나이, 총 구매액, 주문 수, 고객센터 문의 수
- **타겟**: 해지 여부 (0/1)

### 거래 데이터 (Transaction Data)
- **크기**: 50,000건
- **주요 정보**: 거래금액, 판매자, 상태, 프로젝트 유형

## 🚀 주요 기능

### 1️⃣ 대시보드 (Dashboard)
- 평균 해지율, 고위험 고객 수, 분석 대상 현황
- 해지 분포도 및 리스크 스코어 분포
- 고위험 고객 TOP 10 리스트

### 2️⃣ 개별 고객 조회 (Customer Detail)
- 고객ID 선택으로 상세 정보 조회
- 해지 위험도 게이지 차트
- 고객의 거래 이력 분석

### 3️⃣ 세그먼트 분석 (Segment Analysis)
- 연령대별/지역별 해지율 비교
- 세그먼트별 평균 구매액 및 고객 수
- 특성 간 상관관계 히트맵

## 💡 모델 성능

- **알고리즘**: Decision Tree Classifier
- **정확도**: 99.7% (팀 프로젝트 기반)
- **피처 수**: 9개
- **학습 데이터**: 344,299건 (팀 프로젝트)

## 📁 프로젝트 구조

dsadevelop/
├── streamlit_app.py # 메인 애플리케이션
├── requirements.txt # 의존성
├── data/
│ ├── sample_data.py # 더미 데이터 생성
│ ├── customers.csv # 고객 데이터
│ └── transactions.csv # 거래 데이터
├── models/
│ └── predictor.py # 모델 예측 로직
└── README.md

text

## 🎯 사용 방법

### 로컬 실행
1. 저장소 클론
git clone https://github.com/MintPansy/DSAdevelop.git
cd DSAdevelop

2. 필요한 라이브러리 설치
pip install -r requirements.txt

3. Streamlit 앱 실행
streamlit run streamlit_app.py

4. 브라우저에서 http://localhost:8501 접속
text

### 클라우드 접속
[Live Demo Link](배포된 URL)에서 즉시 접속 가능합니다.

## 📈 개발 일정

- **Week 1**: 프로젝트 설계 및 더미 데이터 생성
- **Week 2**: 기본 대시보드 구현
- **Week 3**: 고급 분석 기능 추가
- **Week 4**: 배포 및 최적화

## 🔮 향후 개선 방향

- [ ] 모델 설명 가능성 (SHAP) 추가
- [ ] 시계열 분석 및 추세 예측
- [ ] A/B 테스트 시뮬레이터
- [ ] 배치 분석 및 CSV 다운로드
- [ ] 실시간 데이터 갱신 (DB 연동)

## 📚 참고 자료

- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [Scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [Plotly 시각화](https://plotly.com/python/)

## 👤 작성자

양현준 (MintPansy)
- 데이터 분석 및 머신러닝 엔지니어
- GitHub: [@MintPansy](https://github.com/MintPansy/DSAdevelop)

---

**마지막 업데이트**: 2025년 12월 12일