@echo off
REM =====================================================================
REM tools/env.cmd - cuda-sgemm assembled toolchain bootstrap (no-admin, machine-specific)
REM See results/environment.md section 4. Usage from project root:
REM   cmd /c "call tools\env.cmd && cmake -B build ..."
REM =====================================================================
set CSG_TOOLS=C:\Users\l30086046\csg-tools
call "%CSG_TOOLS%\msvc\devcmd.bat"
set "PATH=%CSG_TOOLS%\cuda-toolkit\bin;%CSG_TOOLS%\git\cmd;%CSG_TOOLS%\cuda\ncu-root\nsight_compute-windows-x86_64-2024.2.0.16-archive\nsight-compute\2024.2.0\target\windows-desktop-win7-x64;%CSG_TOOLS%\cuda\ncu-root\cuda_sanitizer_api-windows-x86_64-12.5.39-archive\compute-sanitizer;%PATH%"
set "CUDA_PATH=%CSG_TOOLS%\cuda-toolkit"
set "CUDACXX=%CSG_TOOLS%\cuda-toolkit\bin\nvcc.exe"
