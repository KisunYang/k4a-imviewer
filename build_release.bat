@echo off
setlocal
cd /d D:\2026\Dev\cursor

if not exist build\CMakeCache.txt (
  cmake -B build -DOpenCV_DIR="D:/2026/Dev/cursor/opencv/opencv/build" -DK4A_ROOT="C:/Program Files/Azure Kinect SDK v1.4.2/sdk"
  if errorlevel 1 exit /b %ERRORLEVEL%
)

cmake --build build --config Release
exit /b %ERRORLEVEL%
