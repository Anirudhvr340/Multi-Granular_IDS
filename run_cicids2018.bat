@echo off
REM =============================================================================
REM CICIDS2018 VALIDATION LAUNCHER: FUSED PROBABILITIES VS. HYBRID QWEN 2.5
REM =============================================================================
REM Usage:
REM   run_cicids2018.bat
REM   run_cicids2018.bat --max-windows 150
REM   run_cicids2018.bat --max-windows 500 --window-seconds 30
REM =============================================================================

"C:\Summer_Internship\venv\Scripts\python.exe" run_cicids2018_comparison.py %*

