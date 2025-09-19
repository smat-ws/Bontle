# Bontle - AI Assistant with Voice Signature Recognition

An intelligent AI assistant that can recognize who is talking using voice signature technology.

## Features

### Core Functionality
- AI-powered conversations using Ollama
- Text-to-Speech (TTS) using OpenAI
- Real-time speech processing
- Weather information
- Spotify integration
- Image searching capabilities

### Voice Signature Recognition 🎤
**NEW**: Advanced speaker identification system that recognizes who is talking based on their unique voice characteristics.

#### Voice Signature Capabilities:
- **Speaker Registration**: Register multiple users with their unique voice signatures
- **Real-time Speaker Identification**: Automatically identify speakers during conversations
- **Speaker-aware Conversations**: Maintain context and personalization based on who is speaking
- **Multi-speaker Support**: Handle conversations with multiple registered users
- **Voice Database Management**: Add, remove, and manage registered speakers

#### How Voice Signature Works:
1. **Feature Extraction**: Analyzes voice characteristics including:
   - Spectral features (frequency distribution, brightness)
   - Prosodic features (pitch, rhythm, energy)
   - Temporal patterns (speaking rate, pauses)
   - Formant features (vocal tract characteristics)

2. **Speaker Registration**: Users record a 5-second voice sample that gets processed and stored as a unique voice signature

3. **Speaker Identification**: During conversations, the system:
   - Records 3-second voice samples
   - Extracts voice features
   - Compares against registered signatures
   - Identifies the speaker with confidence scoring

4. **Conversation Integration**: Identified speakers are automatically tagged in conversation history for personalized responses

## Project Structure

```
Bontle/
├── assist.py              # Main assistant functionality and conversation handling
├── voice_signature.py     # Voice signature recognition system (NEW - modular design)
├── jarvis.py              # Core assistant functionality
├── spot.py                # Spotify integration
├── tools.py               # Utility functions
├── requirements.txt       # Python dependencies
├── voice_signatures.json  # Voice signature database
├── voice_recordings/      # Stored voice samples
└── README.md
```

### Modular Design Benefits:
- **Separation of Concerns**: Voice signature functionality is now isolated in its own module
- **Easy Maintenance**: Voice recognition code is easier to update and debug
- **Reusable**: Voice signature module can be imported and used in other projects
- **Clean Architecture**: Main assistant code is focused on conversation handling
- **Independent Testing**: Voice signature features can be tested independently

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SYSTEM_PROMPT=your_system_prompt_here
   ```

## Usage

### Basic Assistant
```python
from assist import ask_question_memory, TTS

# Ask a question
response = ask_question_memory("Hello, how are you?")
print(response)

# Convert response to speech
TTS(response)
```

### Voice Signature Recognition

#### Using the Voice Signature Module
```python
# Import voice signature functions
from voice_signature import (
    register_new_speaker, 
    identify_current_speaker, 
    get_speaker_info,
    voice_manager
)

# Register a new speaker (will prompt for 5-second recording)
success = register_new_speaker("John Doe", duration=5)

# Identify current speaker (will prompt for 3-second recording)
speaker_name, confidence = identify_current_speaker(duration=3)
print(f"Speaker: {speaker_name} (confidence: {confidence:.2f})")

# Get information about registered speakers
print(get_speaker_info())

# Access the voice manager directly for advanced operations
speakers = voice_manager.list_registered_speakers()
```

#### Speaker-Aware Conversation
```python
from assist import ask_question_memory

# Ask question with automatic speaker identification
response = ask_question_memory("What's the weather like?", identify_speaker=True)
```

#### Manage Voice Database
```python
from assist import list_all_speakers, remove_registered_speaker, get_speaker_info

# List all registered speakers
speakers = list_all_speakers()
print(f"Registered speakers: {speakers}")

# Get speaker information
info = get_speaker_info()
print(info)

# Remove a speaker
remove_registered_speaker("John Doe")
```

### Interactive Assistant Demo
Run the main assistant with voice recognition:

```bash
python assist.py
```

This will launch the Bontle AI Assistant with voice signature capabilities.

### Voice Signature Management (Standalone)
Run the voice signature module independently for setup and testing:

```bash
python voice_signature.py
```

This launches an interactive menu with options to:
1. Register new speakers
2. Identify speakers  
3. List registered speakers
4. Remove speakers
5. Update speaker features
6. Demo conversation with speaker identification

### Command Line Options
```bash
# Run assistant with voice settings menu
python assist.py --voice-settings

# Run main assistant loop (default)
python assist.py
```

## Voice Signature Technical Details

### Feature Extraction
The system extracts 51-dimensional feature vectors from voice samples:
- **Basic Features (8)**: Spectral centroid, rolloff, flux, zero-crossing rate, energy, RMS energy, fundamental frequency, energy variance
- **Mel-scale Features (40)**: Simplified MFCC-like coefficients for frequency content
- **Formant Features (3)**: Energy distribution across frequency bands representing vocal tract characteristics

### Speaker Identification Algorithm
1. **Feature Normalization**: All features are normalized to prevent any single feature from dominating
2. **Cosine Similarity**: Uses cosine distance to compare voice signatures
3. **Confidence Scoring**: Returns similarity scores between 0.0 and 1.0
4. **Threshold-based Classification**: Default threshold of 0.7 for positive identification

### Database Storage
- Voice signatures stored in `voice_signatures.json`
- Audio recordings saved in `voice_recordings/` directory
- Automatic backup and recovery of voice database

## File Structure

```
Bontle/
├── assist.py              # Main assistant with voice signature
├── voice_signatures.json  # Voice signature database
├── voice_recordings/      # Stored voice samples
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
└── README.md              # This file
```

## Dependencies

### Core Dependencies
- `openai` - Text-to-Speech
- `ollama` - AI conversation model
- `pygame` - Audio playback
- `python-dotenv` - Environment variables

### Voice Signature Dependencies
- `numpy` - Numerical computations
- `scipy` - Signal processing
- `soundfile` - Audio file handling
- `librosa` - Audio analysis
- `scikit-learn` - Machine learning utilities
- `pyaudio` - Audio recording

## Configuration

### Environment Variables
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
SYSTEM_PROMPT=Your custom system prompt
```

### Voice Signature Settings
Modify settings in `assist.py`:
```python
# Audio recording parameters
sample_rate = 16000      # Sample rate for recordings
channels = 1             # Mono audio
registration_duration = 5 # Seconds for speaker registration
identification_duration = 3 # Seconds for speaker identification
similarity_threshold = 0.7 # Minimum confidence for positive ID
```

## Advanced Usage

### Custom Voice Features
The voice signature system can be extended with additional features by modifying the `extract_voice_features` method in the `VoiceSignatureManager` class.

### Integration with Speech-to-Text
For full voice-activated conversations, integrate with RealtimeSTT or similar speech recognition systems:

```python
# Example integration (requires RealtimeSTT setup)
def voice_conversation():
    while True:
        # Record audio and convert to text (STT)
        text = speech_to_text()
        
        # Identify speaker from same audio
        speaker, confidence = identify_current_speaker()
        
        # Process with speaker context
        response = ask_question_memory(text, identify_speaker=False)
        
        # Speak response
        TTS(response)
```

## Troubleshooting

### Common Issues

1. **Audio Recording Problems**
   - Ensure microphone permissions are granted
   - Check PyAudio installation: `pip install pyaudio`
   - Verify audio device availability

2. **Feature Extraction Errors**
   - Ensure audio files are valid format
   - Check minimum recording duration (1 second)
   - Verify scipy installation for signal processing

3. **Speaker Recognition Accuracy**
   - Use quiet recording environment
   - Ensure consistent microphone distance
   - Re-register speakers with better quality samples
   - Adjust similarity threshold if needed

### Performance Tips
- Register speakers in quiet environments
- Use consistent microphone setup
- Record for full duration (5 seconds for registration)
- Update features if algorithm improves: use option 5 in interactive demo

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test voice signature functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Changelog

### v2.0.0 - Voice Signature Recognition
- Added complete voice signature recognition system
- Speaker registration and identification
- Speaker-aware conversation tracking
- Interactive demo and management tools
- Comprehensive voice feature extraction
- Voice database management

### v1.0.0 - Initial Release
- Basic AI assistant functionality
- Text-to-Speech integration
- Ollama conversation support

## Contact
Reach out with any feedback or support needs via GitHub or email.
