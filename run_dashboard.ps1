# IT 아웃소싱 고객 해지예측 대시보드 실행 스크립트
# PowerShell 스크립트

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IT 아웃소싱 고객 해지예측 대시보드 실행" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Python 경로 확인
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[확인] Python 버전: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[오류] Python이 설치되어 있지 않거나 PATH에 없습니다." -ForegroundColor Red
    Write-Host "Python을 설치하거나 PATH를 설정해주세요." -ForegroundColor Yellow
    Read-Host "아무 키나 누르면 종료됩니다"
    exit 1
}

# Streamlit 설치 확인
$streamlitCheck = python -m pip show streamlit 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[알림] Streamlit이 설치되어 있지 않습니다. 설치를 시작합니다..." -ForegroundColor Yellow
    python -m pip install streamlit
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[오류] Streamlit 설치에 실패했습니다." -ForegroundColor Red
        Read-Host "아무 키나 누르면 종료됩니다"
        exit 1
    }
    Write-Host "[완료] Streamlit 설치가 완료되었습니다." -ForegroundColor Green
}

# requirements.txt 확인 및 설치
if (Test-Path "requirements.txt") {
    Write-Host "[확인] requirements.txt 파일을 발견했습니다." -ForegroundColor Green
    Write-Host "[알림] 필요한 라이브러리를 설치합니다..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
    Write-Host "[완료] 라이브러리 설치가 완료되었습니다." -ForegroundColor Green
}

# 대시보드 실행
Write-Host ""
Write-Host "대시보드를 실행합니다..." -ForegroundColor Cyan
Write-Host "브라우저가 자동으로 열립니다." -ForegroundColor Cyan
Write-Host ""
Write-Host "종료하려면 Ctrl+C를 누르세요." -ForegroundColor Yellow
Write-Host ""

python -m streamlit run app.py

Read-Host "`n아무 키나 누르면 종료됩니다"

