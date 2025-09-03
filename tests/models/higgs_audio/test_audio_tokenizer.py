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

import unittest

import librosa

from tensorrt_llm.models.higgs_audio.audio_tokenizer import HiggsAudioTokenizer


class TestHiggsAudioTokenizer(unittest.TestCase):
    def test_encode_decode(self):
        tokenizer_dir = "/home/me/TTS/TensorRT-LLM/higgs-audio-v2-generation-3B-base-tokenizer/"
        tokenizer = HiggsAudioTokenizer(tokenizer_dir=tokenizer_dir)

        # Load a sample audio file
        audio_path = "/home/me/TTS/TensorRT-LLM/AussieGirl.wav"
        waveform, sr = librosa.load(audio_path, sr=tokenizer.sampling_rate)

        # Encode the audio
        encoded_output = tokenizer.encode(waveform, sr=sr)
        codes = encoded_output["codes"]

        self.assertIsNotNone(codes)
        self.assertGreater(len(codes), 0)

        # Decode the tokens
        decoded_waveform, decoded_sr = tokenizer.decode(codes)

        self.assertIsNotNone(decoded_waveform)
        self.assertEqual(decoded_sr, tokenizer.sampling_rate)
        self.assertEqual(len(decoded_waveform.shape), 1)
        self.assertGreater(decoded_waveform.shape[0], 0)
