from tensorrt_llm.functional import (
    index_select,
    slice,
)
import torch

yay = torch.arange(8).view(8, 1)
print(yay)
print(slice(yay, [0, 0], [3, 0]))
yay2 = torch.arange(8208).view(8, -1)
print(index_select(yay, 1, 1024))
