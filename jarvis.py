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
        print(current_text)
        if any(hot_word in current_text.lower() for hot_word in hot_words) or skip_hot_word_check:
                    #make sure there is text
                    if current_text:
                        print("User: " + current_text)
                        recorder.stop()
                        #get time
                        current_text = current_text + " " + time.strftime("%Y-m-%d %H-%M-%S")
                        response = assist.ask_question_memory(current_text)
                        print(response)
                        speech = response.split('#')[0]
                        done = assist.TTS(speech)
                        skip_hot_word_check = True if "?" in response else False
                        if len(response.split('#')) > 1:
                            command = response.split('#')[1]
                            tools.parse_command(command)
                        recorder.start()
