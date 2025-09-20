from RealtimeSTT import AudioToTextRecorder
import assist
import time
import tools
import tempfile
import os

# Global variable to store the last recorded audio for speaker identification
last_audio_data = None
last_audio_filename = None

def on_audio_chunk(audio_chunk):
    """Callback to capture audio chunks from RealtimeSTT"""
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

if __name__ == '__main__':
    print("🤖 Bontle Voice Assistant with OpenAI TTS")
    print("🚀 Initializing all models for optimal performance...")
    
    # Note: Using OpenAI TTS as primary (CSM available as fallback)
    print("🎙️ Using OpenAI TTS for high-quality speech synthesis...")
    print("✅ OpenAI TTS ready for instant generation!")
    
    # Initialize RealtimeSTT with model preloading
    print("🎙️ Initializing RealtimeSTT with model preloading...")
    recorder = AudioToTextRecorder(
        spinner=False, 
        model="medium.en",  # You can change to "base.en", "small.en", or "medium.en" for better accuracy
        language="en", 
        post_speech_silence_duration=0.15, 
        silero_sensitivity=0.4, 
        enable_realtime_transcription=False,
        on_recorded_chunk=on_audio_chunk,
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop
    )
    
    print("✅ All models loaded and ready!")
    print("🎯 System optimized for minimal latency")
    
    hot_words = ["bontle","jarvis", "hi"]
    skip_hot_word_check = False
    last_processed_text = ""  # Track last processed text to avoid duplicates
    print("👂 Listening for hotwords...")
    print("💬 Say 'Bontle' or 'Jarvis' to activate")
    
    while True:
        current_text = recorder.text()
        
        # Only print and process if text has changed
        if current_text and current_text != last_processed_text:
            print(current_text)
            
            # Check if hotword detected during TTS playback - interrupt if so
            if assist.is_tts_active() and any(hot_word in current_text.lower() for hot_word in hot_words):
                print("🔄 Hotword detected during TTS - interrupting...")
                assist.interrupt_tts()
                # Continue to process the new command
            
            if any(hot_word in current_text.lower() for hot_word in hot_words) or skip_hot_word_check:
                #make sure there is text
                if current_text:
                    print("User: " + current_text)
                    recorder.stop()
                    
                    # Update last processed text to avoid reprocessing
                    last_processed_text = current_text
                    
                    #get time
                    current_text_with_time = current_text + " " + time.strftime("%Y-m-%d %H-%M-%S")
                    # Enable speaker identification using the captured audio
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
            # Small sleep to prevent busy waiting when no new text
            time.sleep(0.1)
