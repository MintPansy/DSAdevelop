# 빠른 시작 가이드

## "Browse your file system to find a python interpreter" 오류 해결

### 즉시 해결 방법

#### 1단계: Python 설치 확인

PowerShell에서 다음 스크립트를 실행하세요:

```powershell
.\find_python.ps1
```

이 스크립트가 Python의 위치를 찾아줍니다.

#### 2단계: Python이 없는 경우 설치

1. **Python 다운로드**: https://www.python.org/downloads/
2. **설치 시 주의사항**: 
   - ✅ **"Add Python to PATH"** 체크박스를 반드시 선택하세요!
   - ✅ "Install launcher for all users" 선택 (선택사항)

3. **설치 확인**:
   ```powershell
   python --version
   ```

#### 3단계: VS Code에서 Python 인터프리터 선택

**방법 A: 자동 선택**
1. VS Code에서 Python 파일(.py)을 엽니다
2. 우측 하단의 **"Select Python Interpreter"** 클릭
3. Python 버전 선택

**방법 B: 수동 선택**
1. **Ctrl + Shift + P** (또는 **F1**)
2. **"Python: Select Interpreter"** 입력
3. Python 경로 선택

**방법 C: 경로 직접 입력**
만약 Python 경로를 알고 있다면:
1. **Ctrl + Shift + P** → "Python: Select Interpreter"
2. **"Enter interpreter path..."** 선택
3. 경로 입력 (예: `C:\Python311\python.exe`)

### Python 경로 찾기

PowerShell에서:

```powershell
# 방법 1
where python
where py

# 방법 2
Get-Command python
Get-Command py

# 방법 3: 자동 스크립트 실행
.\find_python.ps1
```

### 일반적인 Python 설치 경로

- `C:\Python311\python.exe`
- `C:\Python312\python.exe`
- `C:\Users\사용자명\AppData\Local\Programs\Python\Python311\python.exe`
- `C:\Program Files\Python311\python.exe`

## 설치 후 다음 단계

### 1. 필수 라이브러리 설치

```powershell
python -m pip install -r requirements.txt
```

### 2. 더미데이터 생성 (최초 1회)

```powershell
python data/generate_dummy_data.py
```

### 3. 대시보드 실행

```powershell
python -m streamlit run app.py
```

또는 `run_dashboard.bat` 파일을 더블클릭

## 문제가 계속되면

1. **Python 재설치** (PATH 옵션 체크)
2. **PowerShell 재시작**
3. **VS Code 재시작**
4. **VS Code Python 확장 프로그램 설치 확인**

## 도움말 파일

- `PYTHON_SETUP_GUIDE.md` - 상세한 Python 설정 가이드
- `SETUP_GUIDE.md` - Streamlit 설정 가이드
- `find_python.ps1` - Python 경로 찾기 스크립트

