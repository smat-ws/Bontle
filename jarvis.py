from RealtimeSTT import AudioToTextRecorder
import assist
import time
import tools
import spot

def initialize_system():
    """Initialize all system components"""
    print("🤖 Bontle Voice Assistant Starting...")
    print("=" * 40)
    
    # Initialize Spotify if credentials are provided
    spot.initialize_spotify()
    
    print("🎙️ Initializing speech recognition...")
    return True

if __name__ == '__main__':
    # Initialize system components
    if not initialize_system():
        print("❌ System initialization failed!")
        exit(1)
    
    recorder = AudioToTextRecorder(spinner=False, model="small", language="en"
                                   , post_speech_silence_duration =0.1, silero_sensitivity = 0.4
                                   , enable_realtime_transcription=False)
    hot_words = ["bontle","jarvis","buddy","computer","assistant", "hi","hey"]
    skip_hot_word_check = False
    print("✅ System ready! Say something...")
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
