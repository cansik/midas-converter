#/bin/bash

# setup vars of open vino
[ -z "$INTEL_OPENVINO_DIR" ] && source "/opt/intel/openvino/bin/setupvars.sh"

# install optimizer tools
pushd "/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/"
./install_prerequisites_onnx.sh
popd

# install basic requirements
python -m pip install -r "/opt/intel/openvino/deployment_tools/tools/model_downloader/requirements.in"

# install torch & torch vision
pip install torch torchvision torchaudio

# install pytorch requirements
python -m pip install -r "/opt/intel/openvino/deployment_tools/tools/model_downloader/requirements-pytorch.in"

pip install networkx defusedxml matplotlib opencv-python cmapy
pip install test-generator==0.1.1

# download models
python "/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py" --list models.lst

# convert models
python "/opt/intel/openvino/deployment_tools/tools/model_downloader/converter.py" --list models.lst