@echo off
REM setup vars of open vino
IF "%INTEL_OPENVINO_DIR%"=="" CALL "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python monodepth_convert.py -m public\midasnet\FP32\midasnet.xml -i frames