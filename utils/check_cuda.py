import torch
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Version: ", torch.version.cuda)
import torch
import torch.nn as nn

model = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1).cuda()
input_tensor = torch.randn(1, 3, 224, 224).cuda()
output = model(input_tensor)
print(output.shape)
