# Python 인터프리터 찾기 스크립트
# PowerShell 스크립트

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python 인터프리터 찾기" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 방법 1: where 명령어로 찾기
Write-Host "[방법 1] where 명령어로 찾기" -ForegroundColor Yellow
$pythonPath1 = where.exe python 2>$null
$pyPath1 = where.exe py 2>$null

if ($pythonPath1) {
    Write-Host "✓ python 경로: $pythonPath1" -ForegroundColor Green
} else {
    Write-Host "✗ python을 찾을 수 없습니다." -ForegroundColor Red
}

if ($pyPath1) {
    Write-Host "✓ py 경로: $pyPath1" -ForegroundColor Green
} else {
    Write-Host "✗ py를 찾을 수 없습니다." -ForegroundColor Red
}

Write-Host ""

# 방법 2: Get-Command 사용
Write-Host "[방법 2] Get-Command로 찾기" -ForegroundColor Yellow
try {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        Write-Host "✓ python 경로: $($pythonCmd.Source)" -ForegroundColor Green
        Write-Host "  버전: $(python --version 2>&1)" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ python을 찾을 수 없습니다." -ForegroundColor Red
}

try {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        Write-Host "✓ py 경로: $($pyCmd.Source)" -ForegroundColor Green
        Write-Host "  버전: $(py --version 2>&1)" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ py를 찾을 수 없습니다." -ForegroundColor Red
}

Write-Host ""

# 방법 3: 일반적인 설치 경로 확인
Write-Host "[방법 3] 일반적인 설치 경로 확인" -ForegroundColor Yellow
$commonPaths = @(
    "C:\Python311\python.exe",
    "C:\Python312\python.exe",
    "C:\Python310\python.exe",
    "C:\Program Files\Python311\python.exe",
    "C:\Program Files\Python312\python.exe",
    "C:\Program Files (x86)\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
)

$found = $false
foreach ($path in $commonPaths) {
    if (Test-Path $path) {
        Write-Host "✓ 발견: $path" -ForegroundColor Green
        $found = $true
    }
}

if (-not $found) {
    Write-Host "✗ 일반적인 경로에서 Python을 찾을 수 없습니다." -ForegroundColor Red
}

Write-Host ""

# 방법 4: 레지스트리 확인
Write-Host "[방법 4] 레지스트리에서 찾기" -ForegroundColor Yellow
try {
    $regPaths = Get-ItemProperty -Path "HKLM:\SOFTWARE\Python\PythonCore\*\InstallPath" -ErrorAction SilentlyContinue
    if ($regPaths) {
        foreach ($regPath in $regPaths) {
            $pythonExe = Join-Path $regPath.InstallPath "python.exe"
            if (Test-Path $pythonExe) {
                Write-Host "✓ 발견: $pythonExe" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "✗ 레지스트리에서 Python을 찾을 수 없습니다." -ForegroundColor Red
    }
} catch {
    Write-Host "✗ 레지스트리 확인 중 오류 발생" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "결론" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 최종 확인
$pythonFound = $false
$pythonVersion = $null

try {
    $version = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Python이 정상적으로 설치되어 있습니다!" -ForegroundColor Green
        Write-Host "  버전: $version" -ForegroundColor Green
        Write-Host "  경로: $(Get-Command python).Source" -ForegroundColor Green
        $pythonFound = $true
    }
} catch {
    # 계속 진행
}

if (-not $pythonFound) {
    Write-Host "✗ Python이 설치되어 있지 않거나 PATH에 없습니다." -ForegroundColor Red
    Write-Host ""
    Write-Host "다음 단계:" -ForegroundColor Yellow
    Write-Host "1. Python을 설치하세요: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "2. 설치 시 'Add Python to PATH' 옵션을 체크하세요" -ForegroundColor White
    Write-Host "3. 설치 후 PowerShell을 다시 시작하세요" -ForegroundColor White
}

Write-Host ""
Read-Host "아무 키나 누르면 종료됩니다"

