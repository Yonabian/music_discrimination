import torch
import torchvision
from thop import profile
from model import LSTMFCN

model = LSTMFCN.LSTMFCN(N_time=20,N_Features=92,isOCT=True).cuda()

dummy_input = torch.randn(1, 20, 92).cuda()
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)

model = LSTMFCN.LSTMFCN(N_time=20,N_Features=92).cuda()

dummy_input = torch.randn(1, 20, 92).cuda()
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)