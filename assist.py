from openai import OpenAI
import ollama
import time
import threading
from pygame import mixer
import os
from dotenv import load_dotenv
from datetime import datetime

# Import voice signature functionality from separate module
from voice_signature import (
    voice_manager, 
    register_new_speaker, 
    identify_current_speaker, 
    list_all_speakers, 
    remove_registered_speaker, 
    get_speaker_info, 
    quick_voice_setup, 
    test_voice_signature
)

# Import Parler-TTS functionality
from csm_tts import csm_text_to_speech, initialize_csm_tts

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client (for fallback TTS only) with API key from .env and mixer
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mixer.init()

# Global variable to store conversation history
conversation_history = []

# Global interrupt control for TTS playback
tts_interrupt_flag = threading.Event()
tts_playback_active = threading.Event()

# TTS configuration
USE_LOCAL_CSM_TTS = False  # Set to True to use CSM TTS, False for OpenAI TTS as primary

def ask_question_memory(question, identify_speaker=False, audio_file=None):
    try:
        # Identify speaker if requested and speakers are registered
        speaker_name = "Unknown"
        identify_speaker = False
        if identify_speaker and len(voice_manager.list_registered_speakers()) > 0:
            try:
                if audio_file:
                    # Use provided audio file for speaker identification
                    speaker_name, confidence = voice_manager.identify_speaker(audio_path=audio_file)
                else:
                    # Fallback to recording new audio (legacy behavior)
                    speaker_name, confidence = voice_manager.identify_speaker(duration=2)
                
                if speaker_name != "Unknown":
                    print(f"👤 Speaker: {speaker_name} (confidence: {confidence:.2f})")
                # Don't print anything if speaker is unknown to avoid clutter
            except Exception as e:
                print(f"⚠️ Voice identification failed: {e}")
                speaker_name = "Unknown"
        
        system_message = os.getenv("SYSTEM_PROMPT")

        # Add the new question to the conversation history with speaker info
        question_with_speaker = f"[Speaker: {speaker_name}] {question}" if speaker_name != "Unknown" else question
        conversation_history.append({'role': 'user', 'content': question_with_speaker})
        
        # Include the system message and conversation history in the request
        response = ollama.chat(model='gpt-oss:latest', messages=[
            {'role': 'system', 'content': system_message},
            *conversation_history
        ])
        
        # Get current date and time
        current_datetime = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        
        # Prepend date/time to the response
        formatted_response = f"{current_datetime} {response['message']['content']}"
        
        # Add the AI response to the conversation history (without datetime prefix for context)
        conversation_history.append({'role': 'assistant', 'content': response['message']['content']})
        
        return formatted_response
    except ollama.ResponseError as e:
        print(f"An error occurred: {e}")
        return f"The request failed: {e}"

def generate_tts(sentence, speech_file_path):
    """Generate TTS using OpenAI (fallback only)"""
    response = client.audio.speech.create(model="tts-1", voice="shimmer", input=sentence)
    response.write_to_file(speech_file_path)
    return str(speech_file_path)

def TTS_with_interrupt(text):
    """
    Text-to-Speech function with interrupt capability.
    Can be interrupted by setting the tts_interrupt_flag event.
    """
    global tts_interrupt_flag, tts_playback_active
    
    try:
        # Clear any previous interrupt flags
        tts_interrupt_flag.clear()
        tts_playback_active.set()
        
        print("🎙️ Starting TTS with interrupt capability...")
        
        if USE_LOCAL_CSM_TTS:
            print("🎙️ Using local Sesame CSM TTS...")
            # Generate audio with CSM but don't auto-play, save to default file
            result = csm_text_to_speech(text, play_audio=False, save_file="output.wav")
            
            if "successfully" in result.lower():
                # Check if we were interrupted during generation
                if tts_interrupt_flag.is_set():
                    print("🔄 TTS generation interrupted")
                    tts_playback_active.clear()
                    return "interrupted"
                
                # Play the generated audio with interrupt monitoring
                try:
                    mixer.music.load("output.wav")  # CSM saves to output.wav
                    mixer.music.play()
                    
                    # Monitor for interrupts during playback
                    while mixer.music.get_busy():
                        if tts_interrupt_flag.is_set():
                            print("🔄 TTS playback interrupted")
                            mixer.music.stop()
                            mixer.music.unload()
                            tts_playback_active.clear()
                            return "interrupted"
                        time.sleep(0.1)
                    
                    mixer.music.unload()
                    # Clean up the file
                    if os.path.exists("output.wav"):
                        os.remove("output.wav")
                    tts_playback_active.clear()
                    return "done"
                    
                except Exception as e:
                    print(f"❌ CSM playback error: {e}")
                    # Clean up the file on error
                    if os.path.exists("output.wav"):
                        os.remove("output.wav")
                    raise e
            else:
                raise Exception("CSM TTS generation failed")
                
        else:
            # Use OpenAI TTS with interrupt capability
            print("🔄 Using OpenAI TTS...")
            speech_file_path = generate_tts(text, "speech.mp3")
            
            # Check if interrupted during generation
            if tts_interrupt_flag.is_set():
                print("🔄 TTS generation interrupted")
                if os.path.exists(speech_file_path):
                    os.remove(speech_file_path)
                tts_playback_active.clear()
                return "interrupted"
            
            # Play with interrupt monitoring
            mixer.music.load(speech_file_path)
            mixer.music.play()
            
            while mixer.music.get_busy():
                if tts_interrupt_flag.is_set():
                    print("🔄 TTS playback interrupted")
                    mixer.music.stop()
                    mixer.music.unload()
                    if os.path.exists(speech_file_path):
                        os.remove(speech_file_path)
                    tts_playback_active.clear()
                    return "interrupted"
                time.sleep(0.1)
            
            mixer.music.unload()
            if os.path.exists(speech_file_path):
                os.remove(speech_file_path)
            tts_playback_active.clear()
            return "done"
            
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        tts_playback_active.clear()
        
        # Try fallback if CSM failed
        if USE_LOCAL_CSM_TTS:
            try:
                print("🔄 CSM failed, trying OpenAI fallback...")
                speech_file_path = generate_tts(text, "speech.mp3")
                
                if tts_interrupt_flag.is_set():
                    if os.path.exists(speech_file_path):
                        os.remove(speech_file_path)
                    return "interrupted"
                
                mixer.music.load(speech_file_path)
                mixer.music.play()
                
                while mixer.music.get_busy():
                    if tts_interrupt_flag.is_set():
                        mixer.music.stop()
                        mixer.music.unload()
                        if os.path.exists(speech_file_path):
                            os.remove(speech_file_path)
                        return "interrupted"
                    time.sleep(0.1)
                
                mixer.music.unload()
                if os.path.exists(speech_file_path):
                    os.remove(speech_file_path)
                return "done"
                
            except Exception as e2:
                print(f"❌ Fallback TTS also failed: {e2}")
        
        return "error"

def interrupt_tts():
    """
    Interrupt any ongoing TTS playback.
    Call this function when a hotword is detected during TTS playback.
    """
    global tts_interrupt_flag, tts_playback_active
    
    if tts_playback_active.is_set():
        print("🔄 Interrupting TTS playback...")
        tts_interrupt_flag.set()
        return True
    return False

def is_tts_active():
    """
    Check if TTS is currently active/playing.
    """
    return tts_playback_active.is_set()

def play_sound(file_path):
    mixer.music.load(file_path)
    mixer.music.play()

def TTS(text):
    """
    Text-to-Speech function with CSM as primary and OpenAI as fallback
    """
    try:
        if USE_LOCAL_CSM_TTS:
            print("🎙️ Using local Sesame CSM TTS...")
            result = csm_text_to_speech(text, play_audio=True)
            return "done" if "successfully" in result else "error"
        else:
            # Fallback to OpenAI TTS
            print("🔄 Using OpenAI TTS fallback...")
            speech_file_path = generate_tts(text, "speech.mp3")
            play_sound(speech_file_path)
            while mixer.music.get_busy():
                time.sleep(1)
            mixer.music.unload()
            os.remove(speech_file_path)
            return "done"
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        try:
            # Try OpenAI fallback if CSM fails
            if USE_LOCAL_CSM_TTS:
                print("🔄 CSM failed, trying OpenAI fallback...")
                speech_file_path = generate_tts(text, "speech.mp3")
                play_sound(speech_file_path)
                while mixer.music.get_busy():
                    time.sleep(1)
                mixer.music.unload()
                os.remove(speech_file_path)
                return "done"
        except Exception as e2:
            print(f"❌ Fallback TTS also failed: {e2}")
        return "error"

def main_assistant_loop():
    """Main assistant loop with hotword detection and conversation"""
    global USE_LOCAL_CSM_TTS
    
    print("\n🤖 Bontle AI Assistant")
    print("🎧 Listening for hotwords...")
    print("📢 Say 'Hey Bontle' or 'Bontle' to activate")
    
    # Initialize TTS system
    if USE_LOCAL_CSM_TTS:
        print("✅ Local Sesame CSM TTS enabled")
        print("🚀 Preloading CSM TTS with GPU acceleration...")
        try:
            from csm_tts import preload_csm_model
            preload_csm_model()
            print("⚡ CSM TTS preloaded and ready for instant generation!")
        except Exception as e:
            print(f"⚠️  CSM TTS initialization failed: {e}")
            print("🔄 Will use OpenAI TTS as fallback")
            print("💡 To fix CSM issues:")
            print("   - Ensure stable internet connection")
            print("   - Check HuggingFace token permissions")
            print("   - Try again in a few minutes")
            USE_LOCAL_CSM_TTS = False
    else:
        print("✅ OpenAI TTS enabled as primary")
        print("🎙️ Using OpenAI TTS for high-quality speech synthesis")
        print("💡 CSM TTS available as fallback if needed")
    
    # Check if any speakers are registered
    registered_speakers = voice_manager.list_registered_speakers()
    if not registered_speakers:
        print("\n🎤 Voice Recognition Setup")
        print("👤 No speakers registered yet. Would you like to set up voice recognition?")
        setup_choice = input("🔧 Type 'yes' for quick setup, 'skip' to continue without voice ID: ").strip().lower()
        
        if setup_choice in ['yes', 'y']:
            quick_voice_setup()
        elif setup_choice in ['skip', 's', 'no', 'n']:
            print("⏩ Skipping voice setup. You can set it up later with 'Bontle voice settings'")
        
    print("\n🔧 Commands:")
    print("   'Bontle voice settings' - Manage voice recognition")
    print("   'exit', 'quit', 'bye' - Exit the assistant")
    print("❌ Press Ctrl+C to exit")
    
    try:
        while True:
            # In a real implementation, this would use a hotword detection library
            # For now, we'll simulate with text input
            print("\n" + "="*50)
            user_input = input("💬 Type your message (or 'hotword' to simulate detection): ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            elif 'voice settings' in user_input.lower():
                print("🎤 Opening voice settings...")
                test_voice_signature()
                print("\n🎧 Back to main assistant...")
            elif user_input.lower() == 'hotword' or user_input.startswith('hey') or 'bontle' in user_input.lower():
                if user_input.lower() == 'hotword':
                    print("🔊 Hotword detected! What can I help you with?")
                    question = input("❓ Ask your question: ").strip()
                else:
                    question = user_input
                
                if question:
                    # Check if voice identification is enabled and we have registered speakers
                    enable_voice_id = len(voice_manager.list_registered_speakers()) > 0
                    
                    if enable_voice_id:
                        print("🎤 Processing with voice identification...")
                        response = ask_question_memory(question, identify_speaker=True)
                    else:
                        response = ask_question_memory(question, identify_speaker=False)
                    
                    print(f"🤖 Assistant: {response}")
                    
                    # Optional TTS (uncomment if you want spoken responses)
                    # TTS(response.split('] ', 1)[-1])  # Remove timestamp for TTS
            else:
                print("🔍 No hotword detected. Try saying 'Hey Bontle' or 'Bontle'")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error in main loop: {e}")

if __name__ == "__main__":
    # Main assistant functionality - hotword detection and conversation
    print("🤖 Bontle AI Assistant Starting...")
    print("🎙️ Using Parler-TTS for local high-quality speech synthesis")
    print("Say 'Hey Bontle' or 'Bontle' to start a conversation")
    print("For voice management, say 'Bontle voice settings' or press Ctrl+C and run with --voice-settings")
    
    # Check for voice settings command line argument
    import sys
    if "--voice-settings" in sys.argv:
        test_voice_signature()
    else:
        # Main assistant loop with hotword detection
        main_assistant_loop()

# Export the key functions for use in jarvis.py
__all__ = [
    'ask_question_memory', 
    'TTS', 
    'TTS_with_interrupt', 
    'interrupt_tts', 
    'is_tts_active',
    'conversation_history',
    'voice_manager'
]