import torch
import torch.nn as nn

# s = torch.tensor(0,dtype=torch.float32)
# for i in range(1000): 
#     s += torch.tensor(0.01,dtype=torch.float32)
# print(s)

# s = torch.tensor(0,dtype=torch.float16)
# for i in range(1000):
#     s += torch.tensor(0.01,dtype=torch.float16)
# print(s) 

# s = torch.tensor(0,dtype=torch.float32) 
# for i in range(1000): 
#     s += torch.tensor(0.01,dtype=torch.float16)
# print(s)

# s = torch.tensor(0,dtype=torch.float32) 
# for i in range(1000):
#     x = torch.tensor(0.01,dtype=torch.float16) 
#     s += x.type(torch.float32)
# print(s) 


# class ToyModel(nn.Module):
#     def __init__(self, in_features: int, out_features: int):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, 10, bias=False)
#         self.ln = nn.LayerNorm(10)
#         self.fc2 = nn.Linear(10, out_features, bias=False)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         print(f'fc1 weight : {self.fc1.weight.dtype}')
#         print(f'ln weight : {self.ln.weight.dtype}')
#         print(f'fc2 weight : {self.fc2.weight.dtype}')
#         x = self.relu(self.fc1(x))
#         print('fc1 后', x.dtype)
#         x = self.ln(x)
#         print('ln 后', x.dtype)
#         x = self.fc2(x)
#         print('fc2 后', x.dtype)
#         return x
# criterion = nn.MSELoss()
# model = ToyModel(4, 4).to('cuda')
# x = torch.rand(4, 4, dtype=torch.float32).to('cuda')
# with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#     y = model(x)
#     target = torch.zeros_like(y)
#     loss = criterion(y, target)
#     print('loss dtype:', loss.dtype)
#     loss.backward()
#     print('fc1.grad dtype:', model.fc1.weight.grad.dtype)
#     print('ln.grad dtype:', model.ln.weight.grad.dtype)
#     print('fc2.grad dtype:', model.fc2.weight.grad.dtype)
    
