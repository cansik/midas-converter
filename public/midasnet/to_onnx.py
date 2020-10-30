import os
import glob
import torch
import utils
import cv2
import numpy as np

from torchvision.transforms import Compose
from models.midas_net import MidasNet
from models.transforms import Resize, NormalizeImage, PrepareForNet

import onnx
import onnxruntime

def test_model_accuracy(export_model_name, raw_output, input):    
    ort_session = onnxruntime.InferenceSession(export_model_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)	

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(raw_output), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")		

def export_model(model, input, export_model_name):
    torch.onnx.export(model, input, export_model_name, verbose=False, export_params=True, opset_version=11)	
    onnx_model = onnx.load(export_model_name)    
    onnx.checker.check_model(onnx_model)
    graph_output = onnx.helper.printable_graph(onnx_model.graph)
    with open("graph_output.txt", mode="w") as fout:
        fout.write(graph_output)
		
device = torch.device("cpu")

 # load network
model_path = "model.pt"
model = MidasNet(model_path, non_negative=True)

transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
)

model.to(device)
model.eval()

img = utils.read_image("input/line_up_00.jpg")
img_input = transform({"image": img})["image"]

# compute
#with torch.no_grad():
sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
print("sample type = ", type(sample), ", shape of sample = ", sample.shape)
prediction = model.forward(sample.clone())

export_model_name = "midas.onnx"	

#encoder_script_module = torch.jit.trace(model, img_input)
#encoder_script_module.save(export_model_name)

export_model(model, sample, export_model_name)
test_model_accuracy(export_model_name, prediction, sample)