@echo off
REM setup vars of open vino
CALL "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

python monodepth_webcam.py -m public\midasnet\FP32\midasnet.xml