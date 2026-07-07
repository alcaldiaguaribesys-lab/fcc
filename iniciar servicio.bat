@echo off
TITLE Free Claude Code Launcher
color 0A

echo ====================================================
echo   ARRANCANDO SERVICIOS...
echo ====================================================
cd /d "%~dp0"

:: 1. Backend
start "Servidor Backend" cmd /k "uv run server.py"

timeout /t 5 /nobreak >nul

:: 2. CLI Cliente (Uso npm directo, ignorando python mal configurado)
start "Consola Claude" cmd /k "npx claude"

exit