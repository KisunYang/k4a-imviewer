@echo off
setlocal
set "PATH=D:\2026\Dev\cursor\opencv\opencv\build\x64\vc16\bin;C:\Program Files\Azure Kinect SDK v1.4.2\sdk\windows-desktop\amd64\release\bin;%PATH%"
cd /d D:\2026\Dev\cursor
start "" "D:\2026\Dev\cursor\kinect_view.exe"
