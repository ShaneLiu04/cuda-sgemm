@echo off
REM =====================================================================
REM tools/env.cmd — cuda-sgemm 组装工具链引导（免管理员，机器相关）
REM 详见 results/environment.md §4。用法（在项目根）：
REM   cmd /c "tools\env.cmd && cmake -B build ..."
REM 所有构建/测试/benchmark 命令必须经本脚本设置环境。
REM =====================================================================
set CSG_TOOLS=C:\Users\l30086046\csg-tools
call "%CSG_TOOLS%\msvc\devcmd.bat"
set "PATH=%CSG_TOOLS%\cuda-toolkit\bin;%CSG_TOOLS%\git\cmd;%CSG_TOOLS%\cuda\ncu-root\nsight_compute-windows-x86_64-2024.2.0.16-archive\nsight-compute\2024.2.0\target\windows-desktop-win7-x64;%CSG_TOOLS%\cuda\ncu-root\cuda_sanitizer_api-windows-x86_64-12.5.39-archive\compute-sanitizer;%PATH%"
set "CUDA_PATH=%CSG_TOOLS%\cuda-toolkit"
set "CUDACXX=%CSG_TOOLS%\cuda-toolkit\bin\nvcc.exe"
