@echo off
REM =============================================================================
REM CICIDS2018 FULL 7 GB DATASET VALIDATION LAUNCHER
REM =============================================================================
REM Usage:
REM   validate_full_cicids2018.bat
REM   validate_full_cicids2018.bat --max-windows 150
REM   validate_full_cicids2018.bat --stride 10 --max-windows 200
REM =============================================================================

"C:\Summer_Internship\venv\Scripts\python.exe" validate_full_cicids2018.py %*

