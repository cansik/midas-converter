#/bin/bash

# setup vars of open vino
[ -z "$INTEL_OPENVINO_DIR" ] && source "/opt/intel/openvino/bin/setupvars.sh"

python monodepth_demo.py -m public/midasnet/FP32/midasnet.xml -i images/office.jpg