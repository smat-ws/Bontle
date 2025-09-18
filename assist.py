from openai import OpenAI
import ollama
import time
from pygame import mixer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client (for TTS only) with API key from .env and mixer
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mixer.init()

# Global variable to store conversation history
conversation_history = []

def ask_question_memory(question):
    try:
        system_message = os.getenv("SYSTEM_PROMPT")

        # Add the new question to the conversation history
        conversation_history.append({'role': 'user', 'content': question})
        
        # Include the system message and conversation history in the request
        response = ollama.chat(model='gpt-oss:latest', messages=[
            {'role': 'system', 'content': system_message},
            *conversation_history
        ])
        
        # Add the AI response to the conversation history
        conversation_history.append({'role': 'assistant', 'content': response['message']['content']})
        
        return response['message']['content']
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