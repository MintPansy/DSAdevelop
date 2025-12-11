@echo off
echo ========================================
echo IT 아웃소싱 고객 해지예측 대시보드 실행
echo ========================================
echo.

REM Python 경로 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않거나 PATH에 없습니다.
    echo Python을 설치하거나 PATH를 설정해주세요.
    pause
    exit /b 1
)

REM Streamlit 설치 확인
python -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [알림] Streamlit이 설치되어 있지 않습니다. 설치를 시작합니다...
    python -m pip install streamlit
    if errorlevel 1 (
        echo [오류] Streamlit 설치에 실패했습니다.
        pause
        exit /b 1
    )
)

REM 대시보드 실행
echo 대시보드를 실행합니다...
echo 브라우저가 자동으로 열립니다.
echo.
python -m streamlit run app.py

pause

