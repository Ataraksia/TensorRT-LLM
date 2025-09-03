# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest

from tensorrt_llm.models.higgs_audio.config import (HiggsAudioConfig,
                                                    HiggsAudioEncoderConfig)


class TestHiggsAudioConfig(unittest.TestCase):

    def test_higgs_audio_encoder_config_init(self):
        """Test initialization of HiggsAudioEncoderConfig with default and custom values."""
        # Default initialization
        config = HiggsAudioEncoderConfig()
        self.assertEqual(config.d_model, 1280)
        self.assertEqual(config.encoder_layers, 32)

        # Custom initialization
        config = HiggsAudioEncoderConfig(d_model=1024, encoder_layers=24)
        self.assertEqual(config.d_model, 1024)
        self.assertEqual(config.encoder_layers, 24)

    def test_higgs_audio_config_init(self):
        """Test initialization of HiggsAudioConfig with default and custom values."""
        # Default initialization
        config = HiggsAudioConfig()
        self.assertIsInstance(config.text_config, dict)
        self.assertIsInstance(config.audio_encoder_config,
                              HiggsAudioEncoderConfig)
        self.assertEqual(config.audio_num_codebooks, 8)

        # Custom initialization
        custom_text_config = {"hidden_size": 2048}
        custom_audio_config = {"d_model": 1024}
        config = HiggsAudioConfig(text_config=custom_text_config,
                                  audio_encoder_config=custom_audio_config,
                                  audio_num_codebooks=12)
        self.assertEqual(config.text_config["hidden_size"], 2048)
        self.assertEqual(config.audio_encoder_config.d_model, 1024)
        self.assertEqual(config.audio_num_codebooks, 12)

    def test_serialization_roundtrip(self):
        """Test saving and loading the config to ensure a perfect round-trip."""
        config = HiggsAudioConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            reloaded_config = HiggsAudioConfig.from_pretrained(tmpdir)

            # Convert to dicts for comparison, dropping non-essential keys
            config_dict = config.to_dict()
            reloaded_dict = reloaded_config.to_dict()

            # The 'mapping' key can have a 'rank' subkey that differs, so we remove it
            config_dict.get('mapping', {}).pop('rank', None)
            reloaded_dict.get('mapping', {}).pop('rank', None)

            self.assertEqual(config_dict, reloaded_dict)

    def test_parameter_validation(self):
        """Test that the config validation raises errors for invalid parameters."""
        # Test invalid audio_num_codebooks
        with self.assertRaises(ValueError):
            HiggsAudioConfig(audio_num_codebooks=-1)

        # Test invalid audio_adapter_type
        with self.assertRaises(ValueError):
            HiggsAudioConfig(audio_adapter_type="invalid_adapter")

        # Test invalid token ID
        with self.assertRaises(ValueError):
            HiggsAudioConfig(audio_bos_token_id=-100)

        # Test invalid FFN size
        with self.assertRaises(ValueError):
            HiggsAudioConfig(audio_ffn_hidden_size=0)

    def test_vllm_parity(self):
        """Test for parity with key default values from the vLLM implementation.

        This ensures that our TRT-LLM config aligns with the original model's defaults.
        """
        trt_config = HiggsAudioConfig()
        trt_dict = trt_config.to_dict()

        # Key default values from vLLM's HiggsAudioConfig
        vllm_defaults = {
            "audio_adapter_type": "stack",
            "audio_ffn_hidden_size": 4096,
            "audio_ffn_intermediate_size": 14336,
            "audio_decoder_proj_num_layers": 0,
            "use_delay_pattern": False,
            "audio_codebook_size": 1024,
            "audio_stream_bos_id": 1024,
            "audio_stream_eos_id": 1025,
            "audio_eos_token_id": 128012,
            "audio_out_bos_token_id": 128013,
            "audio_in_token_idx": 128015,
            "audio_out_token_idx": 128016,
            "pad_token_id": 128001,
        }

        # Intentional deviation: vLLM defaults to 12, but our spec is 8.
        # We assert our default is 8 and the vLLM default is different.
        self.assertEqual(trt_dict["audio_num_codebooks"], 8)
        self.assertNotEqual(trt_dict["audio_num_codebooks"], 12)

        # Check all other critical defaults for parity
        for key, value in vllm_defaults.items():
            self.assertEqual(
                trt_dict[key], value,
                f"Mismatch on key '{key}': TRT-LLM default is {trt_dict[key]}, vLLM default is {value}"
            )

    def test_nested_config_creation(self):
        """Test that nested configs (text_config, audio_encoder_config) are created correctly."""
        # Test with dicts
        config = HiggsAudioConfig(
            text_config={"hidden_size": 1024},
            audio_encoder_config={"d_model": 512})
        self.assertIsInstance(config.text_config, dict)
        self.assertEqual(config.text_config["hidden_size"], 1024)
        self.assertIsInstance(config.audio_encoder_config,
                              HiggsAudioEncoderConfig)
        self.assertEqual(config.audio_encoder_config.d_model, 512)

        # Test with pre-instantiated config object
        encoder_conf = HiggsAudioEncoderConfig(d_model=256)
        config = HiggsAudioConfig(audio_encoder_config=encoder_conf)
        self.assertIs(config.audio_encoder_config, encoder_conf)
        self.assertEqual(config.audio_encoder_config.d_model, 256)


if __name__ == "__main__":
    unittest.main()
