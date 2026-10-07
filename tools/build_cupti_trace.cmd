@echo off
REM =====================================================================
REM tools/build_cupti_trace.cmd - AR012 T002 E-B Level-1 collector build
REM NOTE: link.exe LNK1104 on non-ASCII output path (CP936) -> stage link
REM       in an ASCII temp dir, then copy the exe back into build\.
REM Deps : csg-tools toolchain + cupti/cublas component archives
REM Usage: cmd /c "call tools\env.cmd && tools\build_cupti_trace.cmd"
REM =====================================================================
setlocal
set "CSG=C:\Users\l30086046\csg-tools"
set "CUPTI_ARC=%CSG%\cuda\cuda_cupti-windows-x86_64-12.5.39-archive"
set "STAGE=%CSG%\stage-cupti-trace"
if not exist "%CUPTI_ARC%\include\cupti_activity.h" (
    echo [build_cupti_trace] cupti component archive missing: %CUPTI_ARC% & exit /b 1
)
if not exist "%STAGE%" mkdir "%STAGE%"
"%CSG%\cuda-toolkit\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_75 ^
    -I "%~dp0..\include" -I "%CUPTI_ARC%\include" ^
    -L "%CUPTI_ARC%\lib" -lcupti ^
    -L "%CSG%\cuda-toolkit\lib\x64" -lcublas ^
    -o "%STAGE%\cupti_trace.exe" "%~dp0cupti_trace.cu" || exit /b 1
copy /y "%STAGE%\cupti_trace.exe" "%~dp0..\build\" >nul || exit /b 1
copy /y "%CUPTI_ARC%\lib\cupti64_2024.2.0.dll" "%~dp0..\build\" >nul
echo [build_cupti_trace] OK: build\cupti_trace.exe (+cupti dll)
endlocal
