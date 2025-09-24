# import submodules that require registration process
<<<<<<< HEAD
from . import compile, custom_ops, models, shim  # noqa: F401
=======
from . import compile, custom_ops, export, models, shim  # noqa: F401

# import AutoDeploy LLM and LlmArgs
from .llm import *
from .llm_args import *
>>>>>>> upstream/main

try:
    # This will overwrite the AutoModelForCausalLM.from_config to support modelopt quantization
    import modelopt
except ImportError:
    pass
