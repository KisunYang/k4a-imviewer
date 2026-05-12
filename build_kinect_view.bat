@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d D:\2026\Dev\cursor
cl /nologo /EHsc /std:c++17 /O2 /MD ^
  /I"C:\Program Files\Azure Kinect SDK v1.4.2\sdk\include" ^
  /I"D:\2026\Dev\cursor\opencv\opencv\build\include" ^
  main.cpp /Fe:kinect_view.exe ^
  /link ^
  /LIBPATH:"C:\Program Files\Azure Kinect SDK v1.4.2\sdk\windows-desktop\amd64\release\lib" ^
  /LIBPATH:"D:\2026\Dev\cursor\opencv\opencv\build\x64\vc16\lib" ^
  k4a.lib opencv_world490.lib
exit /b %ERRORLEVEL%
