@echo off
setlocal
cd /d D:\2026\Dev\cursor

call build_release.bat
if errorlevel 1 exit /b %ERRORLEVEL%

set "PATH=D:\2026\Dev\cursor\opencv\opencv\build\x64\vc16\bin;C:\Program Files\Azure Kinect SDK v1.4.2\sdk\windows-desktop\amd64\release\bin;%PATH%"
start "" "D:\2026\Dev\cursor\build\Release\k4a_imviewer.exe"
exit /b 0
