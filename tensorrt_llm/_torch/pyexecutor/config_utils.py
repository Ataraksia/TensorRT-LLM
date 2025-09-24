def is_nemotron_hybrid(config):
<<<<<<< HEAD
    if hasattr(config, "hybrid_override_pattern"):
=======
    if hasattr(config, "hybrid_override_pattern"
               ) and config.hybrid_override_pattern is not None and len(
                   config.hybrid_override_pattern) > 0:
>>>>>>> upstream/main
        return True
    return False


def is_mla(config):
<<<<<<< HEAD
    if hasattr(config, "kv_lora_rank"):
        assert hasattr(
            config, "qk_rope_head_dim"
        ), "both of kv_lora_rank and qk_rope_head_dim are required."
=======
    if getattr(config, "kv_lora_rank", None) and getattr(
            config, "qk_rope_head_dim", None):
>>>>>>> upstream/main
        return True
    return False
