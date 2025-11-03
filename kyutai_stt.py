"""
Kyutai STT (Speech-to-Text) Module
High-quality streaming speech recognition using Kyutai STT-2.6B-EN model
"""

import torch
import numpy as np
import tempfile
import os
import time
import soundfile as sf
import pyaudio
import wave
from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

class KyutaiSTT:
    def __init__(self, model_id="kyutai/stt-1b-en_fr"):
        """Initialize the Kyutai STT model"""
        self.model_id = model_id
        self.device = self._get_device()
        self.processor = None
        self.model = None
        self.sample_rate = 24000  # Kyutai STT uses 24kHz
        self.is_loaded = False
        
        # Audio recording parameters
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        print(f"🎙️ Initializing Kyutai STT: {model_id}")
        self._load_model()
    
    def _get_device(self):
        """Determine the best available device"""
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
        """Load the Kyutai STT model and processor"""
        try:
            print(f"🚀 Loading Kyutai STT model on {self.device}...")
            
            # Get Hugging Face token from environment
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token or hf_token == 'your_hf_token_here':
                print("⚠️  No valid Hugging Face token found in .env file")
                print("🔑 Please add your HF token to .env: HUGGINGFACE_TOKEN=your_token_here")
                print("📋 Get your token from: https://huggingface.co/settings/tokens")
                raise ValueError("Missing Hugging Face authentication token")
            
            # Clear GPU cache before loading
            if self.device == "cuda":
                torch.cuda.empty_cache()
                print("🧹 Cleared GPU cache")
            
            # Load processor first (this should work even if model files aren't downloaded)
            print("📦 Loading Kyutai STT processor...")
            self.processor = KyutaiSpeechToTextProcessor.from_pretrained(
                self.model_id,
                token=hf_token,
                trust_remote_code=True
            )
            print("✅ Processor loaded successfully!")
            
            # Now try to load the model - this might fail if large model files aren't downloaded yet
            print(f"🧠 Loading Kyutai STT model on {self.device.upper()}...")
            
            try:
                # Model loading arguments
                model_kwargs = {
                    "token": hf_token,
                    "trust_remote_code": True,
                }
                if self.device == "cuda":
                    model_kwargs.update({
                        "device_map": "cuda:0",
                        "torch_dtype": torch.float16,  # Use FP16 for speed
                        "low_cpu_mem_usage": True,
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
                        "low_cpu_mem_usage": True,
                    })
                    print("🐌 Using CPU with FP32 (consider getting a GPU for better performance)")
                
                self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
                    self.model_id,
                    **model_kwargs
                )
                
                # Warm up the model if on GPU
                if self.device in ["cuda", "mps"]:
                    print("🔥 Warming up Kyutai STT model...")
                    self._warmup_model()
                
                self.is_loaded = True
                print("✅ Kyutai STT model loaded successfully!")
                print(f"💾 Model device: {next(self.model.parameters()).device}")
                print(f"📊 Model dtype: {next(self.model.parameters()).dtype}")
                
            except Exception as model_error:
                print(f"⚠️  Model loading failed: {model_error}")
                print("💡 This usually means the model files are still downloading.")
                print("🔄 Processor is loaded, but model inference will fail until download completes.")
                print("📥 Check download progress with: py download_kyutai_model.py")
                self.model = None
                self.is_loaded = False  # Mark as not fully loaded
        
        except Exception as e:
            print(f"❌ Error loading Kyutai STT model: {e}")
            
            # Better error messages for common issues
            error_str = str(e).lower()
            if "401" in error_str or "unauthorized" in error_str:
                print("🔐 Authentication failed. Please check:")
                print("   1. Your HUGGINGFACE_TOKEN is correct in .env")
                print("   2. Your token has proper permissions")
            elif "connection" in error_str or "timeout" in error_str or "network" in error_str:
                print("🌐 Network connection issue detected:")
                print("   1. Check your internet connection")
                print("   2. Try again in a few minutes")
                print("   3. Consider using a VPN if there are regional restrictions")
            elif "transformers" in error_str:
                print("📦 Transformers version issue:")
                print("   1. Update transformers: pip install transformers>=4.53.0")
                print("   2. Restart Python session after update")
            
            self.is_loaded = False
            raise e
    
    def _warmup_model(self):
        """Warm up the model with a quick generation"""
        try:
            # Create a small dummy audio array
            dummy_audio = np.random.rand(int(self.sample_rate * 0.5))  # 0.5 second
            inputs = self.processor(dummy_audio, return_tensors="pt")
            
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Quick generation to warm up CUDA kernels
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=10)
            
            print("🔥 Model warmup completed!")
            
        except Exception as e:
            print(f"⚠️  Model warmup failed (not critical): {e}")
    
    def transcribe_audio_file(self, audio_path):
        """
        Transcribe audio from a file
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Kyutai STT model not fully loaded - model files may still be downloading. Check: py download_kyutai_model.py")
        
        try:
            # Load audio file
            audio_array, sr = sf.read(audio_path)
            
            # Convert stereo to mono if needed
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to 24kHz if needed
            if sr != self.sample_rate:
                from scipy.signal import resample
                target_length = int(len(audio_array) * self.sample_rate / sr)
                audio_array = resample(audio_array, target_length)
            
            return self.transcribe_audio_array(audio_array)
            
        except Exception as e:
            print(f"❌ Error transcribing audio file {audio_path}: {e}")
            return ""
    
    def transcribe_audio_array(self, audio_array):
        """
        Transcribe audio from numpy array
        
        Args:
            audio_array (np.ndarray): Audio data as numpy array
            
        Returns:
            str: Transcribed text
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Kyutai STT model not fully loaded - model files may still be downloading. Check: py download_kyutai_model.py")
        
        try:
            print(f"🎯 Transcribing audio with Kyutai STT...")
            
            # Normalize audio
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize to [-1, 1] range
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Process audio with Kyutai processor
            inputs = self.processor(audio_array, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                # Clear GPU cache before generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Use mixed precision if on GPU
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        output_tokens = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.8,
                            do_sample=True,
                            top_p=0.9
                        )
                else:
                    output_tokens = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9
                    )
            
            # Decode the generated tokens
            transcribed_text = self.processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
            
            # Clean up the transcription
            transcribed_text = transcribed_text.strip()
            
            print(f"📝 Transcription: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return ""
        finally:
            # Clean up GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def record_and_transcribe(self, duration=5):
        """
        Record audio and transcribe it using Kyutai STT
        
        Args:
            duration (float): Duration to record in seconds
            
        Returns:
            str: Transcribed text
        """
        print(f"🎤 Recording for {duration} seconds...")
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # Open stream
        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        
        # Record audio
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        print("🎯 Processing with Kyutai STT...")
        return self.transcribe_audio_array(audio_array)
    
    def batch_transcribe(self, audio_arrays):
        """
        Transcribe multiple audio arrays in batch for efficiency
        
        Args:
            audio_arrays (list): List of numpy arrays containing audio data
            
        Returns:
            list: List of transcribed texts
        """
        if not self.is_loaded:
            raise RuntimeError("Kyutai STT model not loaded")
        
        try:
            print(f"🎯 Batch transcribing {len(audio_arrays)} audio samples...")
            
            # Normalize all audio arrays
            normalized_arrays = []
            for audio_array in audio_arrays:
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                
                # Normalize to [-1, 1] range
                if np.max(np.abs(audio_array)) > 1.0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                
                normalized_arrays.append(audio_array)
            
            # Process batch with padding
            inputs = self.processor(normalized_arrays, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            
            # Generate transcriptions
            with torch.no_grad():
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    with torch.cuda.amp.autocast():
                        output_tokens = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.8,
                            do_sample=True,
                            top_p=0.9
                        )
                else:
                    output_tokens = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9
                    )
            
            # Decode all transcriptions
            transcriptions = self.processor.batch_decode(output_tokens, skip_special_tokens=True)
            
            # Clean up transcriptions
            cleaned_transcriptions = [text.strip() for text in transcriptions]
            
            print(f"📝 Batch transcription completed: {len(cleaned_transcriptions)} results")
            return cleaned_transcriptions
            
        except Exception as e:
            print(f"❌ Error during batch transcription: {e}")
            return [""] * len(audio_arrays)
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()

# Global instance for easy access
kyutai_stt_instance = None

def initialize_kyutai_stt(model_id="kyutai/stt-2.6b-en-trfs"):
    """Initialize the global Kyutai STT instance"""
    global kyutai_stt_instance
    if kyutai_stt_instance is None:
        print("🚀 Initializing Kyutai STT with GPU acceleration...")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Cleared GPU cache")
        
        kyutai_stt_instance = KyutaiSTT(model_id)
        
        # Display GPU memory usage if available
        if torch.cuda.is_available() and kyutai_stt_instance.device == "cuda":
            torch.cuda.synchronize()
            print(f"🚀 GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
            print(f"💾 GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    return kyutai_stt_instance

def preload_kyutai_stt():
    """Force preload the Kyutai STT model for immediate use"""
    print("⚡ Preloading Kyutai STT model for instant transcription...")
    return initialize_kyutai_stt()

def transcribe_with_kyutai(audio_path=None, audio_array=None, duration=5):
    """
    Convenient function for transcription using Kyutai STT
    
    Args:
        audio_path (str): Path to audio file (optional)
        audio_array (np.ndarray): Audio data as numpy array (optional)
        duration (float): Duration to record if no audio provided
        
    Returns:
        str: Transcribed text
    """
    stt = initialize_kyutai_stt()
    
    if audio_path:
        return stt.transcribe_audio_file(audio_path)
    elif audio_array is not None:
        return stt.transcribe_audio_array(audio_array)
    else:
        return stt.record_and_transcribe(duration)

if __name__ == "__main__":
    # Test the Kyutai STT
    print("🧪 Testing Kyutai STT...")
    
    try:
        stt = KyutaiSTT()
        
        if stt.is_loaded:
            print("\n🎤 Testing recording and transcription...")
            print("💬 Please speak for 3 seconds...")
            
            text = stt.record_and_transcribe(duration=3)
            print(f"✅ Transcription result: '{text}'")
            
            # Test with a batch (just duplicate the same audio for demo)
            print("\n🎯 Testing batch transcription...")
            dummy_audio = np.random.rand(int(stt.sample_rate * 1.0))  # 1 second of noise
            batch_results = stt.batch_transcribe([dummy_audio, dummy_audio])
            print(f"✅ Batch results: {batch_results}")
            
            print("\n🎉 All tests completed successfully!")
        else:
            print("❌ Model failed to load")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure to:")
        print("   - Set your HUGGINGFACE_TOKEN in the .env file")
        print("   - Install transformers>=4.53.0")
        print("   - Have a stable internet connection")