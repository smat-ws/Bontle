#!/usr/bin/env python3
"""
Test script for Kyutai STT functionality
Comprehensive testing of Kyutai STT integration
"""

import sys
import os
import time

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🧪 Testing Dependencies...")
    
    dependencies = [
        ('torch', 'PyTorch for GPU acceleration'),
        ('transformers', 'Hugging Face Transformers >= 4.53.0'),
        ('numpy', 'NumPy for array operations'),
        ('soundfile', 'SoundFile for audio I/O'),
        ('pyaudio', 'PyAudio for recording'),
        ('scipy', 'SciPy for signal processing')
    ]
    
    missing = []
    available = []
    
    for package, description in dependencies:
        try:
            __import__(package)
            available.append(f"✅ {package} - {description}")
        except ImportError:
            missing.append(f"❌ {package} - {description}")
    
    print("\nDependency Status:")
    for item in available:
        print(f"   {item}")
    
    if missing:
        print("\nMissing Dependencies:")
        for item in missing:
            print(f"   {item}")
        return False
    
    return True

def test_transformers_version():
    """Test if transformers version supports Kyutai STT"""
    print("\n🔍 Testing Transformers Version...")
    
    try:
        import transformers
        version = transformers.__version__
        print(f"   Transformers version: {version}")
        
        # Check if version is >= 4.53.0
        version_parts = [int(x) for x in version.split('.')]
        required = [4, 53, 0]
        
        if version_parts >= required:
            print("   ✅ Transformers version supports Kyutai STT")
            return True
        else:
            print(f"   ❌ Transformers version {version} is too old")
            print("   💡 Update with: pip install transformers>=4.53.0")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking transformers: {e}")
        return False

def test_kyutai_import():
    """Test if Kyutai STT classes can be imported"""
    print("\n📦 Testing Kyutai STT Import...")
    
    try:
        from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration
        print("   ✅ Kyutai STT classes imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Cannot import Kyutai STT classes: {e}")
        print("   💡 Make sure transformers >= 4.53.0 is installed")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error importing Kyutai STT: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability for acceleration"""
    print("\n🚀 Testing GPU Availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ CUDA GPU detected: {gpu_name}")
            print(f"   🔥 GPU Memory: {memory:.1f} GB")
            print(f"   📊 GPU Count: {gpu_count}")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("   ✅ Apple MPS detected")
            return 'mps'
        else:
            print("   ⚠️  No GPU detected - will use CPU (slower)")
            return 'cpu'
            
    except Exception as e:
        print(f"   ❌ Error checking GPU: {e}")
        return 'cpu'

def test_huggingface_token():
    """Test Hugging Face token configuration"""
    print("\n🔑 Testing Hugging Face Token...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not token or token == 'your_hf_token_here':
            print("   ❌ No valid Hugging Face token found")
            print("   💡 Add HUGGINGFACE_TOKEN to your .env file")
            print("   📋 Get token from: https://huggingface.co/settings/tokens")
            return False
        
        if token.startswith('hf_'):
            print("   ✅ Valid Hugging Face token format detected")
            return True
        else:
            print("   ⚠️  Token format looks unusual (should start with 'hf_')")
            return True  # Still might work
            
    except Exception as e:
        print(f"   ❌ Error checking token: {e}")
        return False

def test_kyutai_stt_module():
    """Test our Kyutai STT module"""
    print("\n🎯 Testing Kyutai STT Module...")
    
    try:
        # Import our module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from kyutai_stt import KyutaiSTT
        
        print("   ✅ KyutaiSTT module imported successfully")
        return True
        
    except ImportError as e:
        print(f"   ❌ Cannot import KyutaiSTT module: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error testing KyutaiSTT module: {e}")
        return False

def test_stt_config():
    """Test STT configuration system"""
    print("\n⚙️  Testing STT Configuration...")
    
    try:
        from stt_config import stt_config, STTConfig
        
        print(f"   Default Engine: {stt_config.get_active_engine()}")
        print(f"   GPU Enabled: {stt_config.use_gpu}")
        print(f"   Fallback Enabled: {stt_config.fallback_enabled}")
        print(f"   Kyutai Model: {stt_config.kyutai_model}")
        
        print("   ✅ STT configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing STT config: {e}")
        return False

def test_unified_stt():
    """Test unified STT system"""
    print("\n🎛️  Testing Unified STT System...")
    
    try:
        from unified_stt import get_unified_stt, get_stt_status
        
        # This will attempt to initialize the STT system
        print("   Initializing unified STT (this may take a moment)...")
        stt = get_unified_stt()
        
        status = get_stt_status()
        print(f"   Active Engine: {status['active_engine']}")
        print(f"   Kyutai Available: {status['kyutai_available']}")
        print(f"   RealTime Available: {status['realtime_available']}")
        
        if status['kyutai_available'] or status['realtime_available']:
            print("   ✅ At least one STT engine is available")
            return True
        else:
            print("   ❌ No STT engines are available")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing unified STT: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🧪 Kyutai STT Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Transformers Version", test_transformers_version),
        ("Kyutai Import", test_kyutai_import),
        ("GPU Availability", test_gpu_availability),
        ("HuggingFace Token", test_huggingface_token),
        ("KyutaiSTT Module", test_kyutai_stt_module),
        ("STT Configuration", test_stt_config),
        ("Unified STT System", test_unified_stt),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}")
            passed += 1
        else:
            print(f"❌ {test_name}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Kyutai STT should work correctly.")
        return True
    elif passed > failed:
        print("⚠️  Some tests failed but basic functionality should work.")
        print("💡 Check the failed tests and install missing dependencies.")
        return True
    else:
        print("❌ Many tests failed. Kyutai STT may not work properly.")
        print("💡 Please fix the issues before using Kyutai STT.")
        return False

def main():
    """Main test function"""
    success = run_comprehensive_test()
    
    if success:
        print("\n🚀 Ready to use Kyutai STT!")
        print("   Run: python jarvis_enhanced.py")
        print("   Or:  python kyutai_stt.py")
    else:
        print("\n🔧 Please fix the issues and run the test again.")
        print("   Update transformers: pip install transformers>=4.53.0")
        print("   Set HF token in .env: HUGGINGFACE_TOKEN=your_token")
        print("   Install missing dependencies from requirements.txt")

if __name__ == "__main__":
    main()