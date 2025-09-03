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

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig, HiggsAudioEncoderConfig


class TestHiggsAudioConfig(unittest.TestCase):
    def test_higgs_audio_encoder_config_init(self):
        """Test initialization of HiggsAudioEncoderConfig with default and custom values."""
        # Default initialization
        config = HiggsAudioEncoderConfig()
        self.assertEqual(config.d_model, 1280)
        self.assertEqual(config.encoder_layers, 32)

        # Custom initialization
        config = HiggsAudioEncoderConfig(d_model=1040, encoder_layers=24)
        self.assertEqual(config.d_model, 1040)
        self.assertEqual(config.encoder_layers, 24)
        self.assertEqual(config.encoder_attention_heads, 20)  # Should remain default

    def test_higgs_audio_config_init(self):
        """Test initialization of HiggsAudioConfig with default and custom values."""
        # Default initialization
        config = HiggsAudioConfig()
        self.assertIsInstance(config.text_config, dict)
        self.assertIsInstance(config.audio_encoder_config, HiggsAudioEncoderConfig)
        self.assertEqual(config.audio_num_codebooks, 8)

        # Custom initialization
        custom_text_config = {"hidden_size": 2048}
        custom_audio_config = {"d_model": 1040}
        config = HiggsAudioConfig(
            text_config=custom_text_config,
            audio_encoder_config=custom_audio_config,
            audio_num_codebooks=12,
        )
        self.assertEqual(config.text_config["hidden_size"], 2048)
        self.assertEqual(config.audio_encoder_config.d_model, 1040)
        self.assertEqual(config.audio_num_codebooks, 12)

    def test_serialization_roundtrip(self):
        """Test if config can be serialized and deserialized without data loss."""
        config = HiggsAudioConfig()
        config_dict = config.to_dict()
        self.assertIn("model_type", config_dict)
        self.assertEqual(config_dict["model_type"], "higgs_audio")

        # Test from_dict
        new_config = HiggsAudioConfig.from_dict(config_dict)
        self.assertEqual(config.to_dict(), new_config.to_dict())

        # Test JSON serialization
        config_json = config.to_json_string()
        new_config_from_json = HiggsAudioConfig.from_json_string(config_json)
        self.assertEqual(config.to_dict(), new_config_from_json.to_dict())

        # Test saving and loading from pretrained
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded_config = HiggsAudioConfig.from_pretrained(tmpdir)
            self.assertEqual(config.to_dict(), loaded_config.to_dict())

    def test_vllm_parity(self):
        """Test that the default parameters match the vLLM implementation."""
        config = HiggsAudioConfig()

        # Text config defaults
        self.assertEqual(config.text_config["hidden_size"], 3072)
        self.assertEqual(config.text_config["num_hidden_layers"], 28)
        self.assertEqual(config.text_config["num_attention_heads"], 24)
        self.assertEqual(config.text_config["intermediate_size"], 8192)

        # Audio encoder defaults
        self.assertEqual(config.audio_encoder_config.d_model, 1280)
        self.assertEqual(config.audio_encoder_config.encoder_layers, 32)
        self.assertEqual(config.audio_encoder_config.encoder_attention_heads, 20)
        self.assertEqual(config.audio_encoder_config.encoder_ffn_dim, 5120)

        # Top-level audio defaults
        self.assertEqual(config.audio_num_codebooks, 8)
        self.assertEqual(config.audio_codebook_size, 1024)

    def test_parameter_validation(self):
        """Test parameter validation for invalid configurations."""
        # Invalid audio_num_codebooks
        with self.assertRaises(ValueError):
            HiggsAudioConfig(audio_num_codebooks=0)

        # Invalid d_model vs attention heads
        with self.assertRaises(ValueError):
            HiggsAudioEncoderConfig(d_model=1023, encoder_attention_heads=20)

        # Invalid layer numbers
        with self.assertRaises(ValueError):
            HiggsAudioEncoderConfig(encoder_layers=-1)

    def test_nested_config_creation(self):
        """Test that nested configs (text_config, audio_encoder_config) are created correctly."""
        # Test with dicts
        config = HiggsAudioConfig(
            text_config={"hidden_size": 1024}, audio_encoder_config={"d_model": 520}
        )
        self.assertIsInstance(config.text_config, dict)
        self.assertEqual(config.text_config["hidden_size"], 1024)
        self.assertIsInstance(config.audio_encoder_config, HiggsAudioEncoderConfig)
        self.assertEqual(config.audio_encoder_config.d_model, 520)

        # Test with existing config object
        encoder_config = HiggsAudioEncoderConfig(d_model=1040)
        config = HiggsAudioConfig(audio_encoder_config=encoder_config)
        self.assertEqual(config.audio_encoder_config.d_model, 1040)


if __name__ == "__main__":
    unittest.main()
