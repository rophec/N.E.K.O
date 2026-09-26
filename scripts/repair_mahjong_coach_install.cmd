@echo off
chcp 65001 >nul
title Mahjong Coach 安装恢复
echo.
echo 此工具用于修复 N.E.K.O 中残留或重复的 Mahjong Coach 插件目录。
echo 它不会删除用户配置；旧插件文件会移动到可恢复的备份目录。
echo.
echo 请先完全关闭 N.E.K.O，再继续。
pause
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0repair_mahjong_coach_install.ps1" -Apply
set "repair_exit=%ERRORLEVEL%"
echo.
if not "%repair_exit%"=="0" (
  echo 修复未完成，错误码：%repair_exit%
) else (
  echo 修复完成。现在可以重新打开 N.E.K.O 并导入最新版雀魂插件。
)
pause
exit /b %repair_exit%

