from openai import OpenAI
import ollama
import time
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

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client (for TTS only) with API key from .env and mixer
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mixer.init()

# Global variable to store conversation history
conversation_history = []

def ask_question_memory(question, identify_speaker=False):
    try:
        # Identify speaker if requested and speakers are registered
        speaker_name = "Unknown"
        if identify_speaker and len(voice_manager.list_registered_speakers()) > 0:
            try:
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
    response = client.audio.speech.create(model="tts-1", voice="shimmer", input=sentence)
    response.write_to_file(speech_file_path)
    return str(speech_file_path)

def play_sound(file_path):
    mixer.music.load(file_path)
    mixer.music.play()

def TTS(text):
    speech_file_path = generate_tts(text, "speech.mp3")
    play_sound(speech_file_path)
    while mixer.music.get_busy():
        time.sleep(1)
    mixer.music.unload()
    os.remove(speech_file_path)
    return "done"

def main_assistant_loop():
    """Main assistant loop with hotword detection and conversation"""
    print("\n🤖 Bontle AI Assistant")
    print("🎧 Listening for hotwords...")
    print("📢 Say 'Hey Bontle' or 'Bontle' to activate")
    
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
    print("Say 'Hey Bontle' or 'Bontle' to start a conversation")
    print("For voice management, say 'Bontle voice settings' or press Ctrl+C and run with --voice-settings")
    
    # Check for voice settings command line argument
    import sys
    if "--voice-settings" in sys.argv:
        test_voice_signature()
    else:
        # Main assistant loop with hotword detection
        main_assistant_loop()