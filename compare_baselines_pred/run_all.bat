@echo off
cd /d "%~dp0.."
if "%~1"=="" goto :usage
if /i "%~1"=="persistence" goto :persistence
if /i "%~1"=="linear_extrap" goto :linear_extrap
if /i "%~1"=="convlstm" goto :convlstm
if /i "%~1"=="ours" goto :ours
if /i "%~1"=="summary" goto :summary
if /i "%~1"=="all" goto :all
goto :usage
:persistence
echo === [1/4] Persistence ===
python compare_baselines_pred\baselines\persistence\run_persistence.py
goto :eof
:linear_extrap
echo === [2/4] Linear Extrapolation ===
python compare_baselines_pred\baselines\linear_extrap\run_linear_ext.py
goto :eof
:convlstm
echo === [3/4] Pure ConvLSTM ===
python compare_baselines_pred\baselines\convlstm\run_convlstm.py
goto :eof
:ours
echo === [4/4] Ours (GAIN+ConvLSTM) ===
python compare_baselines_pred\baselines\ours\run_ours.py
goto :eof
:summary
echo === [SUMMARY] ===
python compare_baselines_pred\eval_all.py
goto :eof
:all
call "%~f0" persistence
call "%~f0" linear_extrap
call "%~f0" convlstm
call "%~f0" ours
call "%~f0" summary
goto :eof
:usage
echo Usage: run_all.bat [persistence^|linear_extrap^|convlstm^|ours^|summary^|all]
goto :eof
