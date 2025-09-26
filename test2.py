import tensorrt_bindings
from tensorrt_llm.functional import (
    Tensor,
    allgather,
    arange,
    cast,
    categorical_sample,
    constant,
    constant_to_tensor_,
    cumsum,
    expand,
    expand_dims_like,
    gather_last_token_logits,
    int32_array,
    nonzero,
    not_op,
    op_or,
    pad,
    shape,
    softmax,
    unsqueeze,
    view,
    where,
    sum,
    mean,
    concat,
    index_select,
    op_and,
    slice,
)
import torch

yay = torch.arange(8).view(8, 1)
print(yay)
print(slice(yay, [0, 0], [3, 0]))
yay2 = torch.arange(8208).view(8, -1)
print(index_select(yay, 1, 1024))
