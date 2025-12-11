# Streamlit 대시보드 실행 가이드

## 문제 해결: PowerShell에서 streamlit 명령어가 인식되지 않는 경우

### 원인
1. Streamlit이 설치되지 않음
2. Python 가상환경이 활성화되지 않음
3. PATH에 Python 스크립트 경로가 포함되지 않음

## 해결 방법

### 방법 1: Python 모듈로 직접 실행 (권장)

PowerShell에서 다음 명령어를 사용하세요:

```powershell
python -m streamlit run app.py
```

또는

```powershell
py -m streamlit run app.py
```

### 방법 2: Streamlit 설치 확인 및 설치

#### 1단계: Python 설치 확인
```powershell
python --version
# 또는
py --version
```

#### 2단계: pip로 Streamlit 설치
```powershell
python -m pip install streamlit
# 또는
py -m pip install streamlit
```

#### 3단계: 모든 필수 라이브러리 설치
```powershell
python -m pip install -r requirements.txt
# 또는
py -m pip install -r requirements.txt
```

### 방법 3: 가상환경 사용 (권장)

#### 1단계: 가상환경 생성
```powershell
python -m venv venv
# 또는
py -m venv venv
```

#### 2단계: 가상환경 활성화
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# 만약 실행 정책 오류가 나면:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3단계: 라이브러리 설치
```powershell
pip install -r requirements.txt
```

#### 4단계: Streamlit 실행
```powershell
python -m streamlit run app.py
```

### 방법 4: Streamlit 경로 직접 확인

Streamlit이 설치되어 있는지 확인:
```powershell
python -m pip show streamlit
```

설치 경로 확인:
```powershell
python -c "import streamlit; print(streamlit.__file__)"
```

## 전체 실행 순서

### 1. 더미데이터 생성 (최초 1회)
```powershell
python data/generate_dummy_data.py
```

### 2. 대시보드 실행
```powershell
# 방법 A: Python 모듈로 실행 (가장 안정적)
python -m streamlit run app.py

# 방법 B: 직접 실행 (streamlit이 PATH에 있는 경우)
streamlit run app.py
```

## 문제 해결 체크리스트

- [ ] Python이 설치되어 있는가? (`python --version` 확인)
- [ ] Streamlit이 설치되어 있는가? (`python -m pip list | findstr streamlit`)
- [ ] requirements.txt의 모든 라이브러리가 설치되어 있는가?
- [ ] 가상환경을 사용하는 경우 활성화되어 있는가?
- [ ] 올바른 디렉토리에서 명령어를 실행하고 있는가?

## 추가 팁

### PowerShell 실행 정책 오류 해결
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Python 경로 확인
```powershell
where python
# 또는
where py
```

### Streamlit 버전 확인
```powershell
python -m streamlit --version
```

## 대안: 배치 파일 생성

`run_dashboard.bat` 파일을 생성하여 간편하게 실행할 수 있습니다:

```batch
@echo off
python -m streamlit run app.py
pause
```

이 파일을 더블클릭하면 대시보드가 실행됩니다.

