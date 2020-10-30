@echo off

REM setup vars of open vino
CALL "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

REM install optimizer tools
pushd "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\install_prerequisites"
CALL install_prerequisites_onnx.bat
popd

REM install basic requirements
python -m pip install -r "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\requirements.in"

REM install torch & torch vision
pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

REM install pytorch requirements
python -m pip install -r "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\requirements-pytorch.in"

pip install networkx defusedxml matplotlib opencv-python
pip install test-generator==0.1.1

REM download models
python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --list models.lst

REM convert models
python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\converter.py" --list models.lst