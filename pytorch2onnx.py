import torch
from torchvision.models import AlexNet
from torch.hub import load_state_dict_from_url

# 获取PyTorch Module
torch_module = AlexNet()
torch_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
torch_module.load_state_dict(torch_state_dict)
# 设置为eval模式
torch_module.eval()
input_names = ["input_0"]
output_names = ["output_0"]

# 设置模型的输入输出size
x = torch.randn((1, 3, 224, 224))
y = torch.randn((1, 1000))

torch.onnx.export(torch_module, x, './model.onnx', opset_version=11, input_names=input_names,
                  output_names=output_names, dynamic_axes={'input_0': [0], 'output_0': [0]}) 
                  