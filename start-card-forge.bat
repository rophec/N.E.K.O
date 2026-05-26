@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "FORGE_SERVER_ROOT=%PROJECT_ROOT%\local_server\forge_server"
set "CARD_FORGE_FRONTEND_ROOT=%PROJECT_ROOT%\frontend\card-forge"

if not exist "%PROJECT_ROOT%\launcher.py" (
  echo [startup error] N.E.K.O launcher not found:
  echo   "%PROJECT_ROOT%\launcher.py"
  pause
  exit /b 1
)

if not exist "%FORGE_SERVER_ROOT%\server.py" (
  echo [startup error] Forge server not found:
  echo   "%FORGE_SERVER_ROOT%\server.py"
  pause
  exit /b 1
)

if not exist "%CARD_FORGE_FRONTEND_ROOT%\package.json" (
  echo [startup error] card-forge frontend not found:
  echo   "%CARD_FORGE_FRONTEND_ROOT%\package.json"
  pause
  exit /b 1
)

echo ====================================================
echo   N.E.K.O Card Forge - One Click Startup
echo ====================================================
echo Project root: "%PROJECT_ROOT%"
echo.

echo [1/3] Opening N.E.K.O main server window (port 48911)...
start "N.E.K.O Main Server - 48911" "%ComSpec%" /k "cd /d ""%PROJECT_ROOT%"" && uv run .\launcher.py"

timeout /t 3 /nobreak >nul

echo [2/3] Opening forge server window (port 3002)...
start "N.E.K.O Forge Server - 3002" "%ComSpec%" /k "cd /d ""%FORGE_SERVER_ROOT%"" && uv run server.py"

timeout /t 2 /nobreak >nul

echo [3/3] Opening card-forge frontend window (port 5174)...
start "N.E.K.O Card Forge Frontend - 5174" "%ComSpec%" /k "cd /d ""%CARD_FORGE_FRONTEND_ROOT%"" && npm run dev"

echo.
echo ====================================================
echo   Startup commands have been sent to 3 windows.
echo ====================================================
echo URLs:
echo   card-forge dev:   http://127.0.0.1:5174
echo   main app entry:   http://localhost:48911/card_forge
echo   forge API health: http://localhost:3002/health
echo.
echo Keep the three opened command windows running while testing.
pause
