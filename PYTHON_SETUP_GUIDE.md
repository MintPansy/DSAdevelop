# Python 인터프리터 설정 가이드

## 문제: "Browse your file system to find a python interpreter" 오류

이 오류는 VS Code나 다른 IDE가 Python 인터프리터를 찾을 수 없을 때 발생합니다.

## 해결 방법

### 방법 1: Python 설치 확인 및 설치

#### 1단계: Python 설치 여부 확인

PowerShell에서 다음 명령어를 실행하세요:

```powershell
python --version
```

또는

```powershell
py --version
```

**결과가 나오지 않으면 Python이 설치되어 있지 않습니다.**

#### 2단계: Python 설치

1. **Python 공식 웹사이트에서 다운로드**
   - https://www.python.org/downloads/
   - 최신 버전 (Python 3.11 이상 권장) 다운로드
   - **중요**: 설치 시 "Add Python to PATH" 체크박스를 반드시 선택하세요!

2. **설치 확인**
   ```powershell
   python --version
   ```
   버전이 표시되면 설치 완료입니다.

### 방법 2: VS Code에서 Python 인터프리터 선택

#### 자동 선택
1. VS Code를 열고 Python 파일(.py)을 엽니다
2. 우측 하단에 "Select Python Interpreter" 버튼이 나타납니다
3. 클릭하여 Python 인터프리터를 선택합니다

#### 수동 선택
1. **Ctrl + Shift + P** (또는 **F1**)를 눌러 명령 팔레트를 엽니다
2. "Python: Select Interpreter"를 입력하고 선택합니다
3. 설치된 Python 버전을 선택합니다

#### Python 경로 직접 입력
만약 자동으로 찾지 못하면:
1. "Python: Select Interpreter" 선택
2. "Enter interpreter path..." 선택
3. Python 실행 파일 경로 입력:
   - 일반적인 경로:
     - `C:\Python311\python.exe`
     - `C:\Users\사용자명\AppData\Local\Programs\Python\Python311\python.exe`
     - `C:\Program Files\Python311\python.exe`

### 방법 3: Python 경로 찾기

#### Windows에서 Python 경로 찾기

PowerShell에서:

```powershell
# 방법 1: where 명령어
where python
where py

# 방법 2: Get-Command 사용
Get-Command python
Get-Command py

# 방법 3: 레지스트리 확인
Get-ItemProperty -Path "HKLM:\SOFTWARE\Python\PythonCore\*\InstallPath" | Select-Object ExecutablePath
```

#### 일반적인 Python 설치 경로

- `C:\Python311\python.exe`
- `C:\Python312\python.exe`
- `C:\Users\사용자명\AppData\Local\Programs\Python\Python311\python.exe`
- `C:\Program Files\Python311\python.exe`
- `C:\Program Files (x86)\Python311\python.exe`

### 방법 4: PATH 환경 변수 설정

Python이 설치되어 있지만 PATH에 없는 경우:

#### 1단계: Python 설치 경로 확인
위의 방법으로 Python 경로를 찾습니다.

#### 2단계: 환경 변수 설정
1. **시작 메뉴** → "환경 변수" 검색 → "시스템 환경 변수 편집"
2. **환경 변수** 버튼 클릭
3. **시스템 변수**에서 **Path** 선택 → **편집**
4. **새로 만들기** 클릭
5. Python 설치 경로 입력 (예: `C:\Python311`)
6. Python Scripts 경로도 추가 (예: `C:\Python311\Scripts`)
7. **확인** 클릭하여 저장

#### 3단계: PowerShell 재시작
환경 변수 변경 후 PowerShell을 다시 시작합니다.

### 방법 5: 가상환경 사용 (권장)

프로젝트별로 Python 환경을 관리하는 방법:

#### 1단계: 가상환경 생성
```powershell
# Python이 설치되어 있다면
python -m venv venv

# 또는
py -m venv venv
```

#### 2단계: 가상환경 활성화
```powershell
.\venv\Scripts\Activate.ps1
```

#### 3단계: VS Code에서 가상환경 선택
1. VS Code에서 프로젝트 폴더를 엽니다
2. **Ctrl + Shift + P** → "Python: Select Interpreter"
3. `.\venv\Scripts\python.exe` 선택

## 빠른 체크리스트

- [ ] Python이 설치되어 있는가? (`python --version` 확인)
- [ ] Python이 PATH에 포함되어 있는가? (`where python` 확인)
- [ ] VS Code에서 Python 인터프리터를 선택했는가?
- [ ] VS Code Python 확장 프로그램이 설치되어 있는가?

## VS Code Python 확장 프로그램 설치

1. VS Code에서 **확장 프로그램** 탭 열기 (Ctrl + Shift + X)
2. "Python" 검색
3. Microsoft의 "Python" 확장 프로그램 설치

## 문제 해결 후 다음 단계

Python이 정상적으로 설정되면:

1. **필수 라이브러리 설치**
   ```powershell
   python -m pip install -r requirements.txt
   ```

2. **더미데이터 생성**
   ```powershell
   python data/generate_dummy_data.py
   ```

3. **대시보드 실행**
   ```powershell
   python -m streamlit run app.py
   ```

## 추가 도움말

- Python 공식 문서: https://docs.python.org/
- VS Code Python 가이드: https://code.visualstudio.com/docs/python/python-tutorial

