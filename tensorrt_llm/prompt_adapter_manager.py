from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from ._utils import str_dtype_to_torch
from .models.convert_utils import get_model_path, load_state_dict

if TYPE_CHECKING:
    from .runtime import ModelConfig


class PromptAdapterManager:
<<<<<<< HEAD

=======
>>>>>>> upstream/main
    def __init__(self):
        self._uid_counter = 0
        self._uid_to_weights: Dict[str, torch.Tensor] = {}

<<<<<<< HEAD
    def load_from_ckpt(self,
                       model_dirs: List[str],
                       model_config: 'ModelConfig',
                       uids: Optional[List[str]] = None):
=======
    def load_from_ckpt(
        self, model_dirs: List[str], model_config: "ModelConfig", uids: Optional[List[str]] = None
    ):
>>>>>>> upstream/main
        if uids is None:
            uids = [self._generate_uid() for _ in range(len(model_dirs))]
        assert len(uids) == len(model_dirs)

        new_uids, new_model_dirs = [], []
        for uid, model_dir in zip(uids, model_dirs):
            if uid in self._uid_to_weights:
                continue
            new_uids.append(uid)
            new_model_dirs.append(model_dir)

        if len(new_uids) == 0:
            return

        for uid, model_dir in zip(new_uids, new_model_dirs):
<<<<<<< HEAD
            state_dict = load_state_dict(
                get_model_path(model_dir, 'adapter_model'))
            self._uid_to_weights[uid] = state_dict['prompt_embeddings'].to(
                str_dtype_to_torch(model_config.dtype))
=======
            state_dict = load_state_dict(get_model_path(model_dir, "adapter_model"))
            self._uid_to_weights[uid] = state_dict["prompt_embeddings"].to(
                str_dtype_to_torch(model_config.dtype)
            )
>>>>>>> upstream/main

    @property
    def uid_to_weights(self):
        return self._uid_to_weights

    def _generate_uid(self):
        while str(self._uid_counter) in self._uid_to_weights:
            self._uid_counter += 1
        uid = str(self._uid_counter)
        self._uid_counter += 1
        return uid
