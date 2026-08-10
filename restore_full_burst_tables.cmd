@echo off
setlocal
cd /d "%~dp0"
python -m src.utils.restore_full_burst_tables %*
if errorlevel 1 (
    echo.
    echo Restore failed. CarSim tables were not partially updated.
) else (
    echo.
    echo Corrected 100%% burst tables restored successfully.
)
pause
