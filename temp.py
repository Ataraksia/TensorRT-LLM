import torch

one = torch.ones(1, 4192, dtype=torch.int32)
two = torch.ones(1, 172, 4192, dtype=torch.int32)
one_two = one.unsqueeze(1)
result = torch.cat([one_two, two], dim=1)
print(result.shape)  # (1, 173, 4192)
