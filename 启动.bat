@echo off
chcp 65001 >nul 2>&1
title Lama Cleaner++

echo ============================================
echo    Lama Cleaner++ Starting...
echo ============================================
echo.

start /b python "%~dp0lama-cleaner-plusplus\app.py"

:wait_loop
timeout /t 2 /nobreak >nul
powershell -Command "try { $c=New-Object Net.Sockets.TcpClient; $c.Connect('localhost',7860); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 goto wait_loop

start "" http://localhost:7860

echo.
echo Server is running. Close this window to stop.
:keep_alive
timeout /t 60 /nobreak >nul
goto keep_alive
