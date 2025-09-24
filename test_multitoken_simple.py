#!/usr/bin/env python3

"""
Simple test to verify MultiToken DecodingMode integration
"""

import tensorrt_llm
from tensorrt_llm.executor import SamplingConfig, DecodingMode

def test_multitoken_parameter():
    """Test that tokensPerStep parameter is exposed in Python bindings"""
    
    # Test basic SamplingConfig with tokensPerStep
    sampling_config = SamplingConfig(beam_width=1)
    
    # Test setting tokensPerStep
    sampling_config.tokens_per_step = 3
    
    print(f"Created SamplingConfig with tokens_per_step: {sampling_config.tokens_per_step}")
    
    # Test DecodingMode.MultiToken()
    multitoken_mode = DecodingMode.MultiToken()
    print(f"MultiToken mode name: {multitoken_mode.name}")
    print(f"MultiToken mode isMultiToken: {multitoken_mode.isMultiToken()}")
    print(f"MultiToken mode isTopKorTopP: {multitoken_mode.isTopKorTopP()}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_multitoken_parameter()
        if success:
            print("✅ MultiToken parameter test passed!")
        else:
            print("❌ MultiToken parameter test failed!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()