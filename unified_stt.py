"""
Unified STT (Speech-to-Text) Manager
Handles both Kyutai STT and RealTimeSTT with intelligent fallback
"""

import numpy as np
import time
import os
import tempfile
from stt_config import stt_config, STTConfig

class UnifiedSTT:
    def __init__(self):
        """Initialize the unified STT manager"""
        self.config = stt_config
        self.kyutai_stt = None
        self.realtime_stt = None
        self.kyutai_available = False
        self.realtime_available = False
        
        print("🎙️ Initializing Unified STT Manager...")
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available STT engines"""
        
        # Try to initialize Kyutai STT first (default preference)
        if self.config.get_active_engine() == STTConfig.KYUTAI_STT or self.config.fallback_enabled:
            try:
                print("🚀 Attempting to load Kyutai STT...")
                from kyutai_stt import initialize_kyutai_stt
                self.kyutai_stt = initialize_kyutai_stt()
                self.kyutai_available = self.kyutai_stt.is_loaded
                if self.kyutai_available:
                    print("✅ Kyutai STT loaded successfully!")
                else:
                    print("❌ Kyutai STT failed to load")
            except Exception as e:
                print(f"⚠️  Kyutai STT initialization failed: {e}")
                self.kyutai_available = False
        
        # Initialize RealTimeSTT as fallback or primary
        if (self.config.get_active_engine() == STTConfig.REALTIME_STT or 
            self.config.fallback_enabled or 
            not self.kyutai_available):
            try:
                print("🎤 Initializing RealTimeSTT...")
                from RealtimeSTT import AudioToTextRecorder
                
                realtime_config = self.config.get_realtime_config()
                self.realtime_stt = AudioToTextRecorder(**realtime_config)
                self.realtime_available = True
                print("✅ RealTimeSTT initialized successfully!")
            except Exception as e:
                print(f"⚠️  RealTimeSTT initialization failed: {e}")
                self.realtime_available = False
        
        # Determine the best available engine
        self._select_best_engine()
    
    def _select_best_engine(self):
        """Select the best available engine based on preferences and availability"""
        preferred_engine = self.config.get_active_engine()
        
        if preferred_engine == STTConfig.KYUTAI_STT and self.kyutai_available:
            print("🎯 Using Kyutai STT as primary engine")
            self.active_engine = STTConfig.KYUTAI_STT
        elif preferred_engine == STTConfig.REALTIME_STT and self.realtime_available:
            print("🎯 Using RealTimeSTT as primary engine")
            self.active_engine = STTConfig.REALTIME_STT
        elif self.kyutai_available:
            print("🔄 Falling back to Kyutai STT")
            self.active_engine = STTConfig.KYUTAI_STT
        elif self.realtime_available:
            print("🔄 Falling back to RealTimeSTT")
            self.active_engine = STTConfig.REALTIME_STT
        else:
            print("❌ No STT engines available!")
            self.active_engine = None
    
    def transcribe_audio_file(self, audio_path):
        """
        Transcribe audio from a file using the active engine
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        if not self.active_engine:
            return "No STT engine available"
        
        try:
            if self.active_engine == STTConfig.KYUTAI_STT and self.kyutai_available:
                print("🎯 Transcribing with Kyutai STT...")
                return self.kyutai_stt.transcribe_audio_file(audio_path)
            elif self.active_engine == STTConfig.REALTIME_STT and self.realtime_available:
                print("🎯 Transcribing with RealTimeSTT...")
                return self._transcribe_with_realtime(audio_path)
            else:
                return "STT engine not available"
                
        except Exception as e:
            print(f"❌ Primary STT engine failed: {e}")
            
            # Try fallback if enabled
            if self.config.fallback_enabled:
                return self._try_fallback_transcription(audio_path)
            else:
                return f"STT error: {e}"
    
    def transcribe_audio_array(self, audio_array):
        """
        Transcribe audio from numpy array
        
        Args:
            audio_array (np.ndarray): Audio data as numpy array
            
        Returns:
            str: Transcribed text
        """
        if not self.active_engine:
            return "No STT engine available"
        
        try:
            if self.active_engine == STTConfig.KYUTAI_STT and self.kyutai_available:
                print("🎯 Transcribing with Kyutai STT...")
                return self.kyutai_stt.transcribe_audio_array(audio_array)
            elif self.active_engine == STTConfig.REALTIME_STT and self.realtime_available:
                print("🎯 Transcribing with RealTimeSTT...")
                # Save array to temp file for RealTimeSTT
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                import soundfile as sf
                sf.write(temp_path, audio_array, 16000)  # Assume 16kHz for RealTimeSTT
                
                try:
                    result = self._transcribe_with_realtime(temp_path)
                    return result
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                return "STT engine not available"
                
        except Exception as e:
            print(f"❌ Primary STT engine failed: {e}")
            
            # Try fallback if enabled
            if self.config.fallback_enabled:
                return self._try_fallback_transcription_array(audio_array)
            else:
                return f"STT error: {e}"
    
    def _transcribe_with_realtime(self, audio_path):
        """Transcribe audio file using RealTimeSTT"""
        try:
            # RealTimeSTT doesn't have a direct file transcription method
            # We'll need to implement a workaround
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_whisper(audio_data, model=self.config.realtime_model)
                return text
        except Exception as e:
            print(f"❌ RealTimeSTT transcription failed: {e}")
            return ""
    
    def _try_fallback_transcription(self, audio_path):
        """Try fallback STT engine"""
        print("🔄 Trying fallback STT engine...")
        
        if self.active_engine == STTConfig.KYUTAI_STT and self.realtime_available:
            print("🔄 Falling back to RealTimeSTT...")
            return self._transcribe_with_realtime(audio_path)
        elif self.active_engine == STTConfig.REALTIME_STT and self.kyutai_available:
            print("🔄 Falling back to Kyutai STT...")
            return self.kyutai_stt.transcribe_audio_file(audio_path)
        else:
            return "No fallback STT available"
    
    def _try_fallback_transcription_array(self, audio_array):
        """Try fallback STT engine for audio array"""
        print("🔄 Trying fallback STT engine...")
        
        if self.active_engine == STTConfig.KYUTAI_STT and self.realtime_available:
            print("🔄 Falling back to RealTimeSTT...")
            # Save to temp file for RealTimeSTT
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            import soundfile as sf
            sf.write(temp_path, audio_array, 16000)
            
            try:
                return self._transcribe_with_realtime(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        elif self.active_engine == STTConfig.REALTIME_STT and self.kyutai_available:
            print("🔄 Falling back to Kyutai STT...")
            return self.kyutai_stt.transcribe_audio_array(audio_array)
        else:
            return "No fallback STT available"
    
    def get_status(self):
        """Get status of all STT engines"""
        status = {
            'active_engine': self.active_engine,
            'kyutai_available': self.kyutai_available,
            'realtime_available': self.realtime_available,
            'fallback_enabled': self.config.fallback_enabled
        }
        return status
    
    def switch_engine(self, engine):
        """
        Switch to a different STT engine
        
        Args:
            engine (str): Engine to switch to (kyutai or realtime)
        """
        if engine == STTConfig.KYUTAI_STT and self.kyutai_available:
            self.active_engine = STTConfig.KYUTAI_STT
            self.config.set_engine(engine)
            print(f"🔄 Switched to Kyutai STT")
        elif engine == STTConfig.REALTIME_STT and self.realtime_available:
            self.active_engine = STTConfig.REALTIME_STT
            self.config.set_engine(engine)
            print(f"🔄 Switched to RealTimeSTT")
        else:
            print(f"❌ Cannot switch to {engine} - not available")
    
    def get_available_engines(self):
        """Get list of available engines"""
        engines = []
        if self.kyutai_available:
            engines.append(STTConfig.KYUTAI_STT)
        if self.realtime_available:
            engines.append(STTConfig.REALTIME_STT)
        return engines

# Global unified STT instance
unified_stt_instance = None

def get_unified_stt():
    """Get or create the global unified STT instance"""
    global unified_stt_instance
    if unified_stt_instance is None:
        print("🚀 Initializing Unified STT Manager...")
        unified_stt_instance = UnifiedSTT()
    return unified_stt_instance

def transcribe_audio(audio_path=None, audio_array=None):
    """
    Convenient function for audio transcription
    
    Args:
        audio_path (str): Path to audio file (optional)
        audio_array (np.ndarray): Audio data as numpy array (optional)
        
    Returns:
        str: Transcribed text
    """
    stt = get_unified_stt()
    
    if audio_path:
        return stt.transcribe_audio_file(audio_path)
    elif audio_array is not None:
        return stt.transcribe_audio_array(audio_array)
    else:
        return "No audio input provided"

def get_stt_status():
    """Get status of all STT engines"""
    stt = get_unified_stt()
    return stt.get_status()

def switch_stt_engine(engine):
    """Switch STT engine"""
    stt = get_unified_stt()
    stt.switch_engine(engine)

def get_available_stt_engines():
    """Get list of available STT engines"""
    stt = get_unified_stt()
    return stt.get_available_engines()

if __name__ == "__main__":
    # Test the unified STT system
    print("🧪 Testing Unified STT Manager...")
    
    stt = UnifiedSTT()
    
    print(f"\nSTT Status: {stt.get_status()}")
    print(f"Available Engines: {stt.get_available_engines()}")
    
    # Test engine switching
    available_engines = stt.get_available_engines()
    if len(available_engines) > 1:
        print(f"\nTesting engine switch...")
        current = stt.active_engine
        for engine in available_engines:
            if engine != current:
                stt.switch_engine(engine)
                print(f"Switched to: {stt.active_engine}")
                break
        
        # Switch back
        stt.switch_engine(current)
        print(f"Switched back to: {stt.active_engine}")
    
    print("\n🎉 Unified STT test completed!")