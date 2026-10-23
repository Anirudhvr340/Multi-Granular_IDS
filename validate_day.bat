@echo off
REM =============================================================================
REM RUN VALIDATION ON ANY DAY OF CICIDS2018
REM =============================================================================
REM Usage Examples:
REM   validate_day.bat --list-days
REM   validate_day.bat --day 02-14
REM   validate_day.bat --day 02-22 --max-windows 150
REM   validate_day.bat --day 02-21
REM =============================================================================

"C:\Summer_Internship\venv\Scripts\python.exe" validate_cicids2018_day.py %*

