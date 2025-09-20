"""
STT (Speech-to-Text) Configuration Module
Manages STT engine selection and configuration for Bontle AI Assistant
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class STTConfig:
    """Configuration class for STT engines"""
    
    # Available STT engines
    KYUTAI_STT = "kyutai"
    REALTIME_STT = "realtime"
    
    def __init__(self):
        # Default STT engine (can be changed via environment variable)
        self.default_engine = os.getenv('STT_ENGINE', self.KYUTAI_STT)
        self.use_gpu = os.getenv('STT_USE_GPU', 'True').lower() == 'true'
        self.fallback_enabled = os.getenv('STT_FALLBACK_ENABLED', 'True').lower() == 'true'
        
        # Kyutai STT settings
        self.kyutai_model = os.getenv('KYUTAI_MODEL', 'kyutai/stt-2.6b-en-trfs')
        self.kyutai_temperature = float(os.getenv('KYUTAI_TEMPERATURE', '0.8'))
        self.kyutai_max_tokens = int(os.getenv('KYUTAI_MAX_TOKENS', '512'))
        
        # RealTimeSTT settings
        self.realtime_model = os.getenv('REALTIME_STT_MODEL', 'medium.en')
        self.realtime_language = os.getenv('REALTIME_STT_LANGUAGE', 'en')
        self.realtime_silence_duration = float(os.getenv('REALTIME_STT_SILENCE', '0.15'))
        self.realtime_sensitivity = float(os.getenv('REALTIME_STT_SENSITIVITY', '0.4'))
        
        print(f"🎙️ STT Configuration:")
        print(f"   Default Engine: {self.default_engine}")
        print(f"   GPU Enabled: {self.use_gpu}")
        print(f"   Fallback Enabled: {self.fallback_enabled}")
        
    def get_active_engine(self):
        """Get the currently active STT engine"""
        return self.default_engine
    
    def set_engine(self, engine):
        """Set the active STT engine"""
        if engine in [self.KYUTAI_STT, self.REALTIME_STT]:
            self.default_engine = engine
            print(f"🔄 STT engine changed to: {engine}")
        else:
            print(f"❌ Invalid STT engine: {engine}")
    
    def get_kyutai_config(self):
        """Get Kyutai STT configuration"""
        return {
            'model_id': self.kyutai_model,
            'temperature': self.kyutai_temperature,
            'max_tokens': self.kyutai_max_tokens,
            'use_gpu': self.use_gpu
        }
    
    def get_realtime_config(self):
        """Get RealTimeSTT configuration"""
        return {
            'model': self.realtime_model,
            'language': self.realtime_language,
            'post_speech_silence_duration': self.realtime_silence_duration,
            'silero_sensitivity': self.realtime_sensitivity,
            'enable_realtime_transcription': False  # Disabled for better accuracy
        }

# Global configuration instance
stt_config = STTConfig()

def get_stt_config():
    """Get the global STT configuration"""
    return stt_config

def set_stt_engine(engine):
    """Set the global STT engine"""
    stt_config.set_engine(engine)

def get_active_stt_engine():
    """Get the currently active STT engine"""
    return stt_config.get_active_engine()

if __name__ == "__main__":
    # Test the configuration
    print("🧪 Testing STT Configuration...")
    
    config = STTConfig()
    print(f"\nKyutai Config: {config.get_kyutai_config()}")
    print(f"RealTime Config: {config.get_realtime_config()}")
    
    # Test engine switching
    print(f"\nCurrent engine: {config.get_active_engine()}")
    config.set_engine(STTConfig.REALTIME_STT)
    print(f"New engine: {config.get_active_engine()}")
    config.set_engine(STTConfig.KYUTAI_STT)
    print(f"Final engine: {config.get_active_engine()}")