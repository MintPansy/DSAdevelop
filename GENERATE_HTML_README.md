# HTML 대시보드 생성 가이드

## 실행 방법

Streamlit 대시보드의 모든 Plotly 차트를 독립적인 HTML 파일로 변환하려면 다음 명령을 실행하세요:

### 방법 1: Python 직접 실행
```bash
python generate_standalone_html.py
```

### 방법 2: 배치 파일 실행

**Windows CMD에서:**
```bash
generate_html.bat
```

**PowerShell에서 (배치 파일 사용):**
```powershell
.\generate_html.bat
```

또는

```powershell
cmd /c generate_html.bat
```

**PowerShell에서 (PowerShell 스크립트 사용 - 권장):**
```powershell
.\generate_html.ps1
```

> 참고: PowerShell 스크립트 실행이 차단되어 있다면 다음 명령을 먼저 실행하세요:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

## 생성되는 파일

실행이 완료되면 `standalone_dashboard.html` 파일이 프로젝트 루트 디렉토리에 생성됩니다.

## 포함되는 시각화

생성된 HTML 파일에는 다음 6가지 시각화가 포함됩니다:

1. **해지 분포도** - Pie chart (유지/해지 고객 분포)
2. **리스크 스코어 분포** - Histogram (고객 리스크 점수 분포)
3. **특성 중요도** - Horizontal bar chart (모델 특성 중요도)
4. **상관관계 히트맵** - Heatmap (특성 간 상관관계)
5. **세그먼트별 해지율** - Bar chart (구독 유형별 해지율)
6. **고위험 고객 리스트** - Table (리스크 스코어 70점 이상 고객 상위 50명)

## 특징

- ✅ 완전히 독립적인 HTML 파일 (인터넷 연결 필요)
- ✅ Plotly CDN 사용으로 인터랙티브 차트 지원
- ✅ 반응형 디자인 (모바일/태블릿/데스크톱 지원)
- ✅ 전문적인 UI 디자인

## 문제 해결

만약 실행 중 오류가 발생하면:

1. 필요한 패키지가 설치되어 있는지 확인:
   ```bash
   pip install -r requirements.txt
   ```

2. 모델 파일이 존재하는지 확인:
   - `data/model.pkl` 파일이 있어야 합니다
   - 없으면 `python models/train_model.py` 실행

3. Python 버전 확인:
   - Python 3.7 이상 필요
