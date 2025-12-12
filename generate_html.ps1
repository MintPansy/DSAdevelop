# HTML 대시보드 생성 PowerShell 스크립트

# 인코딩 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 스크립트가 있는 디렉토리로 이동
Set-Location -Path $PSScriptRoot

Write-Host "HTML 대시보드 생성 중..." -ForegroundColor Cyan

# Python 스크립트 실행
python generate_standalone_html.py

if ($LASTEXITCODE -eq 0) {
  Write-Host "`n완료되었습니다!" -ForegroundColor Green
}
else {
  Write-Host "`n오류가 발생했습니다." -ForegroundColor Red
  Write-Host "Python이 설치되어 있고 PATH에 등록되어 있는지 확인하세요." -ForegroundColor Yellow
}

Write-Host "`n아무 키나 누르면 종료됩니다..."
Write-Hostl"`n아무l키나 누르면=종료됩니다..."
# 키 입력 대기 (아무 키나 누르면 종료)
[void][System.Console]::ReadKey($true)

