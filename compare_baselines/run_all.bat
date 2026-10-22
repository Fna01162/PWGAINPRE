@echo off
REM ============================================================
REM compare_baselines full experiment runner
REM Usage:
REM   run_all.bat smoke        smoke self-test (2 frames/1 epoch, seconds)
REM   run_all.bat traditional  traditional baselines (no training)
REM   run_all.bat deep         deep baselines (GPU, slowest)
REM   run_all.bat ours         Ours (uses existing checkpoints)
REM   run_all.bat summary      aggregate table1/table2
REM   run_all.bat all          run everything in order
REM ============================================================
set "PY=python"
set "DEV=--device cuda"
cd /d "%~dp0.."

if "%~1"=="" goto :usage
if /i "%~1"=="smoke" goto :smoke
if /i "%~1"=="traditional" goto :traditional
if /i "%~1"=="deep" goto :deep
if /i "%~1"=="ours" goto :ours
if /i "%~1"=="summary" goto :summary
if /i "%~1"=="all" goto :all
goto :usage

:smoke
echo === [SMOKE] deep baseline smoke self-test (CPU, seconds) ===
call :run "%PY% compare_baselines\official_gain\run_gain.py --smoke"
call :run "%PY% compare_baselines\unetpp_cgan\run_unetpp.py --smoke"
call :run "%PY% compare_baselines\dsa_unet\run_dsa.py --smoke"
call :run "%PY% compare_baselines\linear_interp\run_linear.py --smoke"
call :run "%PY% compare_baselines\spatial_knn\run_knn.py --smoke"
call :run "%PY% compare_baselines\ours\run_ours.py --mode sep --smoke"
call :run "%PY% compare_baselines\ours\run_ours.py --mode joint --smoke"
echo === [SMOKE] all passed ===
goto :eof

:traditional
echo === [TRADITIONAL] baselines without training ===
call :run "%PY% compare_baselines\prepare_eval_data.py"
call :run "%PY% compare_baselines\linear_interp\run_linear.py"
call :run "%PY% compare_baselines\spatial_knn\run_knn.py"
call :run "%PY% compare_baselines\no_completion\run_no_completion.py"
echo === [TRADITIONAL] done ===
goto :eof

:deep
echo === [DEEP] train completion net + extension (GPU recommended) ===
call :run "%PY% compare_baselines\official_gain\run_gain.py %DEV%"
call :run "%PY% compare_baselines\unetpp_cgan\run_unetpp.py %DEV%"
call :run "%PY% compare_baselines\dsa_unet\run_dsa.py %DEV%"
echo === [DEEP] done ===
goto :eof

:ours
echo === [OURS] reuse existing checkpoints ===
call :run "%PY% compare_baselines\ours\run_ours.py --mode sep %DEV%"
call :run "%PY% compare_baselines\ours\run_ours.py --mode joint %DEV%"
echo === [OURS] done ===
goto :eof

:summary
echo === [SUMMARY] aggregate table1/table2 ===
call :run "%PY% compare_baselines\eval_all.py"
goto :eof

:all
echo ########## 1/5 smoke ##########
call "%~f0" smoke
if errorlevel 1 goto :fail
echo ########## 2/5 traditional ##########
call "%~f0" traditional
if errorlevel 1 goto :fail
echo ########## 3/5 deep ##########
call "%~f0" deep
if errorlevel 1 goto :fail
echo ########## 4/5 ours ##########
call "%~f0" ours
if errorlevel 1 goto :fail
echo ########## 5/5 summary ##########
call "%~f0" summary
goto :eof

:run
%~1
if errorlevel 1 (
  echo.
  echo *** FAILED: %~1 ***
  exit /b 1
)
exit /b 0

:usage
echo Usage: run_all.bat [smoke^|traditional^|deep^|ours^|summary^|all]
goto :eof

:fail
echo.
echo *** ABORTED ***
exit /b 1
