"""Backend that uses torch.compile only."""

import torch
import torch.nn as nn

<<<<<<< HEAD
=======
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

>>>>>>> upstream/main
from ..compiler import BackendCompiler, BackendRegistry


@BackendRegistry.register("torch-compile")
class TorchCompileCompiler(BackendCompiler):
<<<<<<< HEAD
=======
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ad_logger.info(f"Torch Dynamo cache size limit {torch._dynamo.config.cache_size_limit=}")

>>>>>>> upstream/main
    def compile(self) -> nn.Module:
        """Compile the model using torch.compile."""
        return torch.compile(self.gm, dynamic=True)
