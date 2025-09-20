"""
Sesame CSM Text-to-Speech Module
Local implementation using Hugging Face Transformers
"""

import torch
import os
import time
import tempfile
from transformers import CsmForConditionalGeneration, AutoProcessor
from pygame import mixer
import soundfile as sf
import warnings
import numpy as np
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

class SesameCSM:
    def __init__(self):
        """Initialize the Sesame CSM model locally"""
        self.model_id = "sesame/csm-1b"
        self.device = self._get_device()
        self.processor = None
        self.model = None
        self.sample_rate = 24000  # CSM uses 24kHz sampling rate
        self._load_model()
        
        # Initialize pygame mixer for audio playback
        mixer.init(frequency=self.sample_rate)
        
    def _get_device(self):
        """Determine the best available device - prioritize GPU for speed"""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Apple Metal Performance Shaders (MPS) detected")
            return "mps"
        else:
            print("⚠️  No GPU detected, falling back to CPU (will be slower)")
            return "cpu"
    
    def _load_model(self):
        """Load the Sesame CSM model and processor"""
        try:
            print(f"🚀 Loading Sesame CSM model on {self.device}...")
            
            # Get Hugging Face token from environment
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token or hf_token == 'your_hf_token_here':
                print("⚠️  No valid Hugging Face token found in .env file")
                print("🔑 Please add your HF token to .env: HUGGINGFACE_TOKEN=your_token_here")
                print("📋 Get your token from: https://huggingface.co/settings/tokens")
                print("🔐 Make sure you've accepted the conditions at: https://huggingface.co/sesame/csm-1b")
                raise ValueError("Missing Hugging Face authentication token")
            
            # Load processor and model with authentication
            print("📦 Loading CSM processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                token=hf_token,
                trust_remote_code=True
            )
            
            print(f"🧠 Loading CSM model on {self.device.upper()}...")
            
            # Force GPU usage and optimize settings
            model_kwargs = {
                "token": hf_token,
                "trust_remote_code": True,
            }
            
            if self.device == "cuda":
                model_kwargs.update({
                    "device_map": "cuda:0",  # Force specific GPU
                    "torch_dtype": torch.float16,  # Use FP16 for speed
                    "low_cpu_mem_usage": True,     # Optimize memory usage
                })
                print("⚡ Using CUDA with FP16 optimization")
            elif self.device == "mps":
                model_kwargs.update({
                    "device_map": "mps",
                    "torch_dtype": torch.float16,
                })
                print("🍎 Using Apple MPS with FP16 optimization")
            else:
                model_kwargs.update({
                    "device_map": "cpu",
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True,  # Optimize CPU memory usage
                })
                print("🐌 Using CPU with FP32 (consider getting a GPU for better performance)")
                print("🧠 Optimizing for CPU inference...")
            
            self.model = CsmForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            # Warm up the model with a test generation if on GPU
            if self.device in ["cuda", "mps"]:
                print("🔥 Warming up CSM model...")
                self._warmup_model()
            
            print("✅ Sesame CSM model loaded successfully!")
            print(f"💾 Model device: {next(self.model.parameters()).device}")
            print(f"📊 Model dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            print(f"❌ Error loading Sesame CSM model: {e}")
            
            # Better error messages for common issues
            error_str = str(e).lower()
            if "401" in error_str or "unauthorized" in error_str:
                print("🔐 Authentication failed. Please check:")
                print("   1. Your HUGGINGFACE_TOKEN is correct in .env")
                print("   2. You've accepted the model conditions at: https://huggingface.co/sesame/csm-1b")
                print("   3. Your token has proper permissions")
            elif "connection" in error_str or "timeout" in error_str or "network" in error_str:
                print("🌐 Network connection issue detected:")
                print("   1. Check your internet connection")
                print("   2. Try again in a few minutes (HuggingFace might be busy)")
                print("   3. Consider downloading the model manually if this persists")
                print("   4. You can also try using a VPN if there are regional restrictions")
            elif "disk" in error_str or "space" in error_str:
                print("💾 Disk space issue:")
                print("   1. Free up disk space (model requires ~3GB)")
                print("   2. Check available space in your cache directory")
            else:
                print("🔍 For other issues, check:")
                print("   1. Your PyTorch installation supports the model")
                print("   2. All dependencies are up to date")
                print("   3. Restart and try again")
            raise e
    
    def _warmup_model(self):
        """Warm up the model with a quick generation to prepare for fast inference"""
        try:
            warmup_text = "[0]Test"
            inputs = self.processor(warmup_text, add_special_tokens=True)
            
            # Move inputs to device
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.device)
            else:
                for key in inputs:
                    if hasattr(inputs[key], 'to'):
                        inputs[key] = inputs[key].to(self.device)
            
            # Quick generation to warm up CUDA kernels
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs, 
                    output_audio=True,
                    max_new_tokens=64,  # Short warmup
                    temperature=0.8,
                    do_sample=True
                )
            print("🔥 Model warmup completed!")
            
        except Exception as e:
            print(f"⚠️  Model warmup failed (not critical): {e}")
    
    def generate_audio(self, text, speaker_id="0", temperature=0.9):
        """
        Generate audio from text using Sesame CSM
        
        Args:
            text (str): Text to convert to speech
            speaker_id (str): Speaker ID for voice consistency ("0", "1", etc.)
            temperature (float): Controls randomness in generation (0.1-1.0)
            
        Returns:
            torch.Tensor: Generated audio tensor
        """
        if not self.model or not self.processor:
            raise RuntimeError("Sesame CSM model not loaded")
        
        try:
            # Clean and validate text
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")
            
            # Limit text length to prevent memory issues
            max_length = 500  # Adjust based on your needs
            if len(text) > max_length:
                print(f"⚠️  Text too long ({len(text)} chars), truncating to {max_length}")
                text = text[:max_length].rsplit('.', 1)[0] + '.'  # Try to end at sentence
            
            # Format text with speaker ID according to CSM format
            formatted_text = f"[{speaker_id}]{text}"
            print(f"🎯 Formatted text: {formatted_text[:100]}{'...' if len(formatted_text) > 100 else ''}")
            
            # Method 1: Simple text processing
            inputs = self.processor(
                formatted_text, 
                add_special_tokens=True
            ).to(self.device)
            
            # Alternative method using conversation format (more advanced)
            # conversation = [
            #     {"role": speaker_id, "content": [{"type": "text", "text": text}]},
            # ]
            # inputs = self.processor.apply_chat_template(
            #     conversation,
            #     tokenize=True,
            #     return_dict=True,
            # ).to(self.device)
            
            # Generate audio with CSM - optimized for GPU performance
            generation_kwargs = {
                "output_audio": True,
                "max_new_tokens": 2048,
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "use_cache": True,  # Enable KV cache for speed
            }
            
            # Add GPU-specific optimizations
            if self.device == "cuda":
                generation_kwargs.update({
                    "pad_token_id": self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None,
                })
            
            with torch.no_grad():
                # Clear GPU cache before generation to prevent OOM
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                try:
                    # Use torch.cuda.amp for mixed precision if on GPU
                    if self.device == "cuda":
                        with torch.cuda.amp.autocast():
                            audio = self.model.generate(**inputs, **generation_kwargs)
                    else:
                        audio = self.model.generate(**inputs, **generation_kwargs)
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("🚨 GPU out of memory! Clearing cache and retrying...")
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        # Retry with simpler settings
                        generation_kwargs.update({
                            "max_new_tokens": 1024,  # Reduce token count
                            "use_cache": False,      # Disable cache to save memory
                        })
                        # Second attempt
                        if self.device == "cuda":
                            with torch.cuda.amp.autocast():
                                audio = self.model.generate(**inputs, **generation_kwargs)
                        else:
                            audio = self.model.generate(**inputs, **generation_kwargs)
                    else:
                        raise e
            
            return audio
            
        except Exception as e:
            print(f"❌ Error generating audio with CSM: {e}")
            return None
        finally:
            # Clean up GPU memory after generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def save_audio(self, audio_tensor, filename):
        """Save audio tensor to file using CSM processor"""
        if audio_tensor is not None:
            try:
                self.processor.save_audio(audio_tensor, filename)
                return filename
            except Exception as e:
                print(f"❌ Error saving audio: {e}")
                # Fallback: convert to numpy and save with soundfile
                if hasattr(audio_tensor, 'cpu'):
                    audio_array = audio_tensor.cpu().numpy()
                    sf.write(filename, audio_array, self.sample_rate)
                    return filename
        return None
    
    def play_audio(self, audio_tensor):
        """Play audio tensor directly"""
        if audio_tensor is None:
            return
            
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Save audio to temporary file
            self.save_audio(audio_tensor, temp_filename)
            
            # Play audio
            mixer.music.load(temp_filename)
            mixer.music.play()
            
            # Wait for playback to complete
            while mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            mixer.music.unload()
            os.unlink(temp_filename)
            
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
    
    def text_to_speech(self, text, speaker_id="1", temperature=0.7, play_audio=True, save_file=None):
        """
        Convert text to speech and optionally play or save
        
        Args:
            text (str): Text to convert to speech
            speaker_id (str): Speaker ID for voice consistency
            temperature (float): Controls randomness in generation
            play_audio (bool): Whether to play the audio immediately
            save_file (str): Optional filename to save audio
            
        Returns:
            str: Status message
        """
        try:
            print(f"🎙️ Generating speech with CSM for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"   Speaker ID: {speaker_id}, Temperature: {temperature}")
            
            # Generate audio
            audio = self.generate_audio(text, speaker_id, temperature)
            
            if audio is None:
                return "Failed to generate audio"
            
            # Save if requested
            if save_file:
                self.save_audio(audio, save_file)
                print(f"💾 Audio saved to: {save_file}")
            
            # Play if requested
            if play_audio:
                self.play_audio(audio)
            
            return "CSM speech generation completed successfully"
            
        except Exception as e:
            error_msg = f"Error in CSM text_to_speech: {e}"
            print(f"❌ {error_msg}")
            return error_msg

# Global instance for easy access
csm_instance = None

def initialize_csm_tts():
    """Initialize the global Sesame CSM instance with GPU optimization"""
    global csm_instance
    if csm_instance is None:
        print("🚀 Initializing Sesame CSM TTS with GPU acceleration...")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Cleared GPU cache")
        
        csm_instance = SesameCSM()
        
        # Optimize GPU memory if available
        if torch.cuda.is_available() and csm_instance.device == "cuda":
            torch.cuda.synchronize()
            print(f"🚀 GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
            print(f"💾 GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
        
    return csm_instance

def preload_csm_model():
    """Force preload the CSM model for immediate use"""
    print("⚡ Preloading Sesame CSM model for instant TTS...")
    return initialize_csm_tts()

def csm_text_to_speech(text, speaker_id="0", temperature=0.9, play_audio=True, save_file=None):
    """
    Convenient function for text-to-speech conversion using Sesame CSM
    
    Args:
        text (str): Text to convert to speech
        speaker_id (str): Speaker ID for voice consistency
        temperature (float): Controls randomness in generation
        play_audio (bool): Whether to play the audio immediately
        save_file (str): Optional filename to save audio
        
    Returns:
        str: Status message
    """
    tts = initialize_csm_tts()
    return tts.text_to_speech(text, speaker_id, temperature, play_audio, save_file)

# Backward compatibility aliases
parler_text_to_speech = csm_text_to_speech
initialize_parler_tts = initialize_csm_tts

if __name__ == "__main__":
    # Test the Sesame CSM TTS
    print("🧪 Testing Sesame CSM TTS...")
    try:
        tts = SesameCSM()
        
        # Test basic text-to-speech
        print("\n🎵 Testing basic speech generation...")
        result = tts.text_to_speech(
            "Hello! This is a test of the Sesame CSM text-to-speech system. The quality should be excellent for local generation."
        )
        print(f"✅ Basic test result: {result}")
        
        # Test with different speaker IDs
        print("\n🎭 Testing different speaker voices...")
        for speaker_id in ["0", "1", "2"]:
            result = tts.text_to_speech(
                f"This is speaker {speaker_id} testing the voice synthesis.",
                speaker_id=speaker_id,
                play_audio=False,  # Don't play all of them
                save_file=f"test_speaker_{speaker_id}.wav"
            )
            print(f"🎤 Speaker {speaker_id} test: {result}")
        
        # Test temperature variations
        print("\n🌡️  Testing temperature variations...")
        for temp in [0.3, 0.7, 1.0]:
            result = tts.text_to_speech(
                f"Temperature test at {temp}",
                temperature=temp,
                play_audio=False,
                save_file=f"test_temp_{temp}.wav"
            )
            print(f"🔥 Temperature {temp} test: {result}")
            
        print("\n🎉 All tests completed successfully!")
        print("📁 Audio files saved for review")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure to set your HUGGINGFACE_TOKEN in the .env file")
        print("🔐 And accept the model conditions at: https://huggingface.co/sesame/csm-1b")