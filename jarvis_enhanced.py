"""
Enhanced Jarvis with Unified STT Support
Supports both Kyutai STT and RealTimeSTT with intelligent fallback
"""

import assist
import time
import tools
import tempfile
import os
import numpy as np
import spot
from stt_config import stt_config, STTConfig
from unified_stt import get_unified_stt, get_stt_status, switch_stt_engine, get_available_stt_engines

# Global variables for audio capture and STT management
last_audio_data = None
last_audio_filename = None
unified_stt = None

def initialize_system():
    """Initialize all system components"""
    print("🤖 Bontle Voice Assistant - Enhanced Edition")
    print("=" * 50)
    
    # Initialize Spotify if credentials are provided
    print("\n🎵 Initializing Spotify Integration...")
    spot.initialize_spotify()
    
    return True

def initialize_stt_system():
    """Initialize the STT system with user choice"""
    global unified_stt
    
    print("\n🎙️ Initializing Speech-to-Text System...")
    print("=" * 50)
    
    # Initialize unified STT
    unified_stt = get_unified_stt()
    status = get_stt_status()
    available_engines = get_available_stt_engines()
    
    print(f"\n🎙️ STT System Status:")
    print(f"   Available Engines: {', '.join(available_engines)}")
    print(f"   Active Engine: {status['active_engine']}")
    print(f"   Kyutai STT: {'✅ Available' if status['kyutai_available'] else '❌ Not Available'}")
    print(f"   RealTime STT: {'✅ Available' if status['realtime_available'] else '❌ Not Available'}")
    print(f"   Fallback Enabled: {'✅ Yes' if status['fallback_enabled'] else '❌ No'}")
    
    # Allow user to choose engine if multiple are available
    if len(available_engines) > 1:
        print(f"\n🔧 Multiple STT engines available!")
        print("   1. Kyutai STT (High-quality, GPU-accelerated, 2.5s delay)")
        print("   2. RealTime STT (Real-time, lower delay, good quality)")
        print("   3. Use default configuration")
        
        choice = input("\n💭 Choose STT engine (1-3) or press Enter for default: ").strip()
        
        if choice == "1" and STTConfig.KYUTAI_STT in available_engines:
            switch_stt_engine(STTConfig.KYUTAI_STT)
            print("🎯 Using Kyutai STT as primary engine")
        elif choice == "2" and STTConfig.REALTIME_STT in available_engines:
            switch_stt_engine(STTConfig.REALTIME_STT)
            print("🎯 Using RealTime STT as primary engine")
        else:
            print(f"🎯 Using default engine: {status['active_engine']}")
    
    return unified_stt

def transcribe_audio_with_fallback(audio_file=None, audio_array=None):
    """
    Transcribe audio using the unified STT system with fallback
    
    Args:
        audio_file (str): Path to audio file
        audio_array (np.ndarray): Audio data as numpy array
        
    Returns:
        str: Transcribed text
    """
    global unified_stt
    
    if not unified_stt:
        unified_stt = get_unified_stt()
    
    try:
        if audio_file:
            return unified_stt.transcribe_audio_file(audio_file)
        elif audio_array is not None:
            return unified_stt.transcribe_audio_array(audio_array)
        else:
            return ""
    except Exception as e:
        print(f"❌ STT transcription failed: {e}")
        return ""

def on_audio_chunk(audio_chunk):
    """Callback to capture audio chunks (for RealTimeSTT compatibility)"""
    global last_audio_data
    if last_audio_data is None:
        last_audio_data = audio_chunk
    else:
        last_audio_data += audio_chunk

def on_recording_start():
    """Reset audio data when recording starts"""
    global last_audio_data
    last_audio_data = None

def on_recording_stop():
    """Save captured audio to file when recording stops"""
    global last_audio_data, last_audio_filename
    if last_audio_data is not None:
        # Create temporary file for the captured audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            last_audio_filename = temp_file.name
        
        # Save the audio data to file
        try:
            import wave
            with wave.open(last_audio_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(last_audio_data)
        except Exception as e:
            print(f"Error saving audio: {e}")
            last_audio_filename = None

def run_realtime_stt_mode():
    """Run with RealTimeSTT for continuous listening"""
    from RealtimeSTT import AudioToTextRecorder
    
    print("🎤 Starting RealTime STT mode...")
    print("👂 Listening for hotwords...")
    print("💬 Say 'Bontle' or 'Jarvis' to activate")
    print("🙏 Say 'Thank you' to interrupt TTS and activate listening")
    
    # Initialize RealTimeSTT with callbacks
    realtime_config = stt_config.get_realtime_config()
    realtime_config.update({
        'on_recorded_chunk': on_audio_chunk,
        'on_recording_start': on_recording_start,
        'on_recording_stop': on_recording_stop,
        'spinner': False
    })
    
    recorder = AudioToTextRecorder(**realtime_config)
    
    hot_words = ["bontle", "jarvis", "hi"]
    interrupt_words = ["thank you", "thanks"]
    skip_hot_word_check = False
    last_processed_text = ""
    
    while True:
        current_text = recorder.text()
        
        # Only process if text has changed
        if current_text and current_text != last_processed_text:
            print(current_text)
            current_text_lower = current_text.lower()
            
            # Check for interrupt words during TTS playback
            if assist.is_tts_active() and any(interrupt_word in current_text_lower for interrupt_word in interrupt_words):
                print("🙏 'Thank you' detected - interrupting TTS and activating listening...")
                assist.interrupt_tts()
                skip_hot_word_check = True  # Put the assistant in listening mode
                recorder.stop()
                recorder.start()
                continue
            
            # Check if hotword detected during TTS playback - interrupt if so
            if assist.is_tts_active() and any(hot_word in current_text_lower for hot_word in hot_words):
                print("🔄 Hotword detected during TTS - interrupting...")
                assist.interrupt_tts()
            
            if any(hot_word in current_text_lower for hot_word in hot_words) or skip_hot_word_check:
                if current_text:
                    print("User: " + current_text)
                    recorder.stop()
                    
                    last_processed_text = current_text
                    current_text_with_time = current_text + " " + time.strftime("%Y-m-%d %H-%M-%S")
                    
                    # Use unified STT for speaker identification if we have better audio
                    response = assist.ask_question_memory(current_text_with_time, identify_speaker=True, audio_file=last_audio_filename)
                    print(response)
                    speech = response.split('#')[0]
                    
                    # Use TTS with interrupt capability
                    done = assist.TTS_with_interrupt(speech)
                    
                    if done == "interrupted":
                        print("🔄 TTS was interrupted - processing new command immediately")
                    
                    skip_hot_word_check = True if "?" in response else False
                    if len(response.split('#')) > 1:
                        command = response.split('#')[1]
                        tools.parse_command(command)
                    
                    # Clean up temporary audio file
                    if last_audio_filename and os.path.exists(last_audio_filename):
                        try:
                            os.remove(last_audio_filename)
                        except:
                            pass
                        last_audio_filename = None
                    
                    recorder.start()
        else:
            time.sleep(0.1)

def run_kyutai_stt_mode():
    """Run with Kyutai STT for high-quality transcription"""
    print("🎯 Starting Kyutai STT mode...")
    print("🎤 Press Enter to start recording, or type 'quit' to exit")
    print("🙏 Say 'Thank you' to interrupt TTS and activate listening")
    
    hot_words = ["bontle", "jarvis", "hi"]
    interrupt_words = ["thank you", "thanks"]
    
    while True:
        user_input = input("\n💬 Press Enter to speak (or 'quit'/'switch' to change mode): ").strip().lower()
        
        if user_input in ['quit', 'exit', 'q']:
            break
        elif user_input in ['switch', 'change']:
            return 'switch'
        elif user_input == '' or user_input in hot_words:
            print("🎤 Recording for 5 seconds... Speak now!")
            
            try:
                # Record and transcribe with Kyutai STT
                current_text = transcribe_audio_with_fallback()
                
                if current_text:
                    print(f"📝 Transcribed: {current_text}")
                    current_text_lower = current_text.lower()
                    
                    # Check for interrupt words during TTS playback
                    if assist.is_tts_active() and any(interrupt_word in current_text_lower for interrupt_word in interrupt_words):
                        print("🙏 'Thank you' detected - interrupting TTS and ready for next command...")
                        assist.interrupt_tts()
                        continue  # Go back to listening immediately
                    
                    # Check if hotword detected during TTS playback
                    if assist.is_tts_active() and any(hot_word in current_text_lower for hot_word in hot_words):
                        print("🔄 Hotword detected during TTS - interrupting...")
                        assist.interrupt_tts()
                    
                    print("User: " + current_text)
                    
                    if any(hot_word in current_text_lower for hot_word in hot_words):
                        current_text_with_time = current_text + " " + time.strftime("%Y-m-%d %H-%M-%S")
                        
                        # Process with assistant
                        response = assist.ask_question_memory(current_text_with_time, identify_speaker=False)
                        print(response)
                        speech = response.split('#')[0]
                        
                        # Use TTS with interrupt capability
                        done = assist.TTS_with_interrupt(speech)
                        
                        if done == "interrupted":
                            print("🔄 TTS was interrupted - ready for next command")
                        
                        if len(response.split('#')) > 1:
                            command = response.split('#')[1]
                            tools.parse_command(command)
                    else:
                        print("🔍 No hotword detected. Try saying 'Bontle' or 'Jarvis'")
                else:
                    print("❌ No speech detected. Try speaking louder or closer to the microphone.")
                    
            except Exception as e:
                print(f"❌ Error during recording/transcription: {e}")
        else:
            print("🔍 Say 'Bontle' or 'Jarvis' to activate the assistant")
    
    return 'quit'

def main():
    """Main function to run the enhanced assistant"""
    print("🚀 Starting Enhanced Bontle Assistant...")
    
    # Initialize system components (Spotify, etc.)
    if not initialize_system():
        print("❌ System initialization failed!")
        return
    
    # Initialize STT system
    unified_stt = initialize_stt_system()
    status = get_stt_status()
    
    if not status['kyutai_available'] and not status['realtime_available']:
        print("❌ No STT engines are available!")
        print("💡 Please check your dependencies and configuration")
        return
    
    print("\n✅ All systems initialized successfully!")
    print("\n🎯 Assistant Commands:")
    print("   💬 Hotwords: 'Bontle', 'Jarvis', 'Hi'")
    print("   � Interrupt TTS: 'Thank you' (puts assistant in listening mode)")
    print("   �🔄 Switch STT: Say 'switch engine' or 'change engine'")
    print("   🎵 Music: 'play', 'pause', 'skip', 'previous'")
    print("   ❌ Exit: Say 'exit', 'quit', or press Ctrl+C")
    
    current_mode = status['active_engine']
    
    try:
        while True:
            print(f"\n🎙️ Current STT Engine: {current_mode}")
            
            if current_mode == STTConfig.REALTIME_STT:
                result = run_realtime_stt_mode()
            elif current_mode == STTConfig.KYUTAI_STT:
                result = run_kyutai_stt_mode()
            else:
                print("❌ No active STT engine")
                break
            
            if result == 'quit':
                break
            elif result == 'switch':
                # Switch to the other available engine
                available = get_available_stt_engines()
                if len(available) > 1:
                    new_engine = STTConfig.REALTIME_STT if current_mode == STTConfig.KYUTAI_STT else STTConfig.KYUTAI_STT
                    if new_engine in available:
                        switch_stt_engine(new_engine)
                        current_mode = new_engine
                        print(f"🔄 Switched to {new_engine}")
                    else:
                        print(f"❌ {new_engine} not available")
                else:
                    print("❌ Only one STT engine is available")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == '__main__':
    main()