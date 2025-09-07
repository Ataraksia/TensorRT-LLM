import torch

one = torch.ones(1, 4192, dtype=torch.int32)
two = torch.ones(1, 172, 4192, dtype=torch.int32) * 2

three = one.unsqueeze(-1) + two
print(three.shape)
