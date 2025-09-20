# Bontle - Advanced AI Assistant with Dual STT, Voice Signatures & Interrupt TTS

An intelligent AI assistant featuring dual speech recognition engines (Kyutai STT + RealTimeSTT), advanced voice recognition, interruptible text-to-speech, and seamless real-time interaction capabilities.

## ✨ Latest Features

### 🎯 **NEW: Kyutai STT Integration** (v4.0.0)
Revolutionary dual STT system with state-of-the-art Kyutai STT-2.6B and intelligent fallback:
- **Kyutai STT-2.6B-EN**: State-of-the-art speech recognition with 2.6B parameters
- **GPU Acceleration**: Optimized for NVIDIA GPUs and Apple Silicon (MPS)
- **Intelligent Fallback**: Automatic fallback to RealTimeSTT for reliability
- **Engine Selection**: Choose between high-quality and real-time transcription
- **Unified Management**: Single interface for multiple STT engines
- **Advanced Configuration**: Comprehensive settings for optimal performance

### 🔄 **TTS with Interrupt** (v3.0.0)
Revolutionary interruptible text-to-speech system that allows natural conversation flow:
- **Instant Interruption**: Say hotwords during TTS playback to interrupt and give new commands immediately
- **Natural Conversation**: No more waiting for responses to finish - just like talking to a human
- **Seamless Integration**: Uses same audio from hotword detection for speaker identification
- **Thread-Safe**: Robust interrupt handling with proper resource cleanup
- **Dual TTS Support**: Primary Sesame CSM (local, GPU-accelerated) with OpenAI fallback

### 🎙️ **Enhanced STT System**
- **Kyutai STT-2.6B-EN**: High-quality, GPU-accelerated speech recognition with 2.5s delay
- **RealTimeSTT Integration**: Real-time speech recognition with minimal delay
- **Intelligent Fallback**: Automatic engine switching for reliability
- **GPU Optimization**: CUDA and MPS acceleration for maximum performance
- **Engine Selection**: Runtime switching between quality and speed modes
- **Batch Processing**: Efficient processing of multiple audio files

### � **Enhanced TTS System**
- **Sesame CSM TTS**: Local, high-quality text-to-speech with GPU acceleration
- **Model Preloading**: Instant response times with preloaded models
- **Smart Fallback**: Automatic fallback to OpenAI TTS if CSM fails
- **Interrupt Capability**: All TTS engines support immediate interruption

### 🎯 **Real-time Speech Integration**
- **Dual STT Engines**: Kyutai STT (quality) + RealTimeSTT (speed)
- **Audio Chunk Reuse**: Uses hotword detection audio for speaker identification (no additional delays)
- **GPU Optimization**: All models load on GPU for maximum performance
- **Minimal Latency**: Optimized for instant response times

## Core Features

### AI Conversation System
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
├── assist.py                    # Main assistant with TTS interrupt functionality
├── jarvis.py                    # Real-time speech processing with RealTimeSTT
├── jarvis_enhanced.py           # Enhanced assistant with dual STT engines
├── kyutai_stt.py               # Kyutai STT-2.6B implementation with GPU optimization
├── stt_config.py               # STT engine configuration management
├── unified_stt.py              # Unified STT manager with intelligent fallback
├── csm_tts.py                  # Sesame CSM TTS implementation with GPU optimization
├── voice_signature.py          # Voice signature recognition system
├── spot.py                     # Spotify integration
├── tools.py                    # Utility functions
├── test_interrupt.py           # TTS interrupt functionality testing
├── test_kyutai_stt.py          # Comprehensive Kyutai STT testing
├── requirements.txt            # Python dependencies
├── voice_signatures.json       # Voice signature database
├── voice_recordings/           # Stored voice samples
├── KYUTAI_STT_SETUP_GUIDE.md   # Comprehensive Kyutai STT setup guide
├── TTS_INTERRUPT_GUIDE.md      # Complete TTS interrupt documentation
├── CSM_SETUP_GUIDE.md         # CSM TTS setup and configuration
├── GPU_OPTIMIZATION_SUMMARY.md # GPU optimization details
└── README.md
```

### Architecture Highlights:
- **Modular Design**: Each component is separate and focused on specific functionality
- **Dual STT System**: Kyutai STT (quality) + RealTimeSTT (speed) with intelligent selection
- **Thread-Safe**: Interrupt system uses proper threading mechanisms
- **GPU Optimized**: All models prioritize GPU usage for maximum performance
- **Fallback Support**: Robust error handling with automatic fallbacks
- **Real-time Processing**: Optimized for minimal latency and instant responses

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

4. **For Kyutai STT support**, ensure you have the latest transformers:
   ```bash
   pip install transformers>=4.53.0
   ```

5. Create a `.env` file with your API keys and STT configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   HUGGINGFACE_TOKEN=your_hf_token_here
   SYSTEM_PROMPT=your_system_prompt_here
   OLLAMA_BASE_URL=http://localhost:11434  # Optional: if using custom Ollama setup
   
   # STT Configuration
   STT_ENGINE=kyutai                    # Options: kyutai, realtime
   STT_USE_GPU=True                     # Enable GPU acceleration
   STT_FALLBACK_ENABLED=True            # Enable automatic fallback
   ```

6. Install PyTorch with CUDA support (for GPU acceleration):
   ```bash
   # For CUDA 11.8
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

7. **Test your Kyutai STT setup**:
   ```bash
   python test_kyutai_stt.py
   ```

## Quick Start

### Enhanced Assistant with Dual STT (Recommended)
```bash
# Start the enhanced assistant with Kyutai STT + RealTimeSTT
python jarvis_enhanced.py
```

**Features:**
- 🎯 **Dual STT Engines**: Choose between Kyutai STT (quality) and RealTimeSTT (speed)
- 🔄 **Intelligent Fallback**: Automatic engine switching for reliability
- ⚡ **GPU Acceleration**: Optimized Kyutai STT on CUDA/MPS
- 🎙️ **Interrupt TTS**: Say hotwords during playback to interrupt immediately
- 👤 **Speaker Recognition**: Automatic speaker identification using captured audio
- 🚀 **Model Preloading**: All models preloaded for instant responses

### Real-time Assistant (RealTimeSTT Primary)
```bash
# Start the main assistant with continuous hotword detection
python jarvis.py
```

**Features:**
- 🎙️ Continuous listening for hotwords ("Bontle", "Jarvis", "Hi")
- 🔄 Interrupt TTS by saying hotwords during playback
- 👤 Automatic speaker identification using hotword audio
- ⚡ GPU-accelerated CSM TTS for high-quality speech
- 🚀 Model preloading for instant responses

### Interactive Assistant (Text-based)
```bash
# Run text-based assistant with voice recognition setup
python assist.py
```

## TTS with Interrupt System

### How It Works
1. **Natural Conversation**: Say "Bontle" to start a conversation
2. **Interrupt Anytime**: Say "Bontle" again during TTS playback to interrupt and give new commands
3. **Seamless Transition**: No delays - immediate response to interruption
4. **Smart Audio Reuse**: Uses the same audio for both hotword detection and speaker identification

### Example Usage
```python
from assist import TTS_with_interrupt, interrupt_tts, is_tts_active

# Start interruptible TTS
result = TTS_with_interrupt("This is a long message that can be interrupted...")

# Check status
if result == "interrupted":
    print("TTS was interrupted by user")
elif result == "done":
    print("TTS completed successfully")
elif result == "error":
    print("TTS failed, check logs")

# Programmatically interrupt TTS
if is_tts_active():
    interrupt_tts()
```

### Testing Interrupt Functionality
```bash
# Run comprehensive interrupt tests
python test_interrupt.py
```

## Usage

### Basic Assistant
```python
from assist import TTS_with_interrupt, TTS, csm_text_to_speech

# Interruptible TTS (recommended for real-time applications)
result = TTS_with_interrupt("Hello, this message can be interrupted.")

# Standard TTS with CSM + OpenAI fallback
TTS("Hello, this is a standard TTS message.")

# Direct CSM TTS usage
csm_text_to_speech("Hello from Sesame CSM!", play_audio=True)
```

### Dual STT System
```python
from unified_stt import get_unified_stt, switch_stt_engine, get_stt_status

# Get STT status
status = get_stt_status()
print(f"Active Engine: {status['active_engine']}")
print(f"Available Engines: {status['kyutai_available']}, {status['realtime_available']}")

# Switch engines at runtime
switch_stt_engine('kyutai')    # High quality, 2.5s delay
switch_stt_engine('realtime')  # Real-time, minimal delay

# Transcribe audio with automatic engine selection
stt = get_unified_stt()
text = stt.transcribe_audio_file("recording.wav")
```

### Kyutai STT Direct Usage
```python
from kyutai_stt import transcribe_with_kyutai

# High-quality transcription
text = transcribe_with_kyutai(audio_path="recording.wav")

# Or with numpy array
text = transcribe_with_kyutai(audio_array=audio_data)

# Batch processing for efficiency
from kyutai_stt import initialize_kyutai_stt
stt = initialize_kyutai_stt()
results = stt.batch_transcribe([audio1, audio2, audio3])
```

### Basic Assistant Functions
```python
from assist import ask_question_memory

# Ask a question with speaker identification
response = ask_question_memory("Hello, how are you?", identify_speaker=True)
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
- `openai` - OpenAI API for fallback TTS
- `ollama` - Local AI conversation model
- `pygame` - Audio playback and mixing
- `python-dotenv` - Environment variable management

### STT and Speech Processing
- `torch` - PyTorch for GPU acceleration
- `torchaudio` - Audio processing with PyTorch
- `transformers>=4.53.0` - Hugging Face models (Kyutai STT support)
- `soundfile` - Audio file I/O
- `RealtimeSTT` - Real-time speech recognition
- `scipy` - Signal processing for audio resampling

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

### v4.0.0 - Kyutai STT Integration & Dual Engine System (September 2025)
- 🎯 **Kyutai STT-2.6B-EN**: State-of-the-art speech recognition with 2.6B parameters
- 🔄 **Dual STT System**: Intelligent choice between Kyutai STT and RealTimeSTT
- ⚡ **GPU Acceleration**: CUDA and Apple MPS optimization for Kyutai STT
- 🎛️ **Unified STT Manager**: Single interface for multiple STT engines
- 🔄 **Intelligent Fallback**: Automatic engine switching for reliability
- ⚙️ **Advanced Configuration**: Comprehensive STT settings and optimization
- 🧪 **Testing Suite**: Complete test framework for STT validation
- 📚 **Enhanced Documentation**: Detailed setup guides and usage examples

### v3.0.0 - TTS with Interrupt & Real-time Speech (September 2025)
- ✨ **TTS Interrupt System**: Revolutionary interruptible text-to-speech
- 🎙️ **Sesame CSM TTS**: Local, GPU-accelerated high-quality TTS
- 🔄 **Real-time Integration**: Seamless RealtimeSTT integration with audio reuse
- ⚡ **GPU Optimization**: All models preloaded on GPU for instant responses
- 🎯 **Hotword Detection**: Continuous listening with interrupt capability
- 🧵 **Thread-Safe Design**: Robust interrupt handling with proper cleanup
- 📊 **Smart Fallback**: CSM → OpenAI TTS with interrupt support in both
- 🚀 **Performance**: Minimal latency with optimized model loading

### v2.0.0 - Voice Signature Recognition (Previous)
- 👤 Added complete voice signature recognition system
- 🎤 Speaker registration and identification
- 💬 Speaker-aware conversation tracking
- 🔧 Interactive demo and management tools
- 📈 Comprehensive voice feature extraction
- 📁 Voice database management

### v1.0.0 - Initial Release
- Basic AI assistant functionality
- Text-to-Speech integration
- Ollama conversation support

## Performance Benchmarks

### STT Engine Comparison (GPU)
| Engine | Accuracy | Speed | GPU Usage | Real-time | Best Use Case |
|--------|----------|-------|-----------|-----------|---------------|
| **Kyutai STT** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | High-quality transcription |
| **RealTimeSTT** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Interactive conversations |

### STT Response Times (GPU)
- **Kyutai STT Processing**: ~2.5 seconds (with 2.5s model delay)
- **RealTimeSTT Processing**: ~0.2-0.5 seconds
- **Engine Switching**: <0.1 seconds
- **Model Loading Time**: ~5-10 seconds (one-time at startup)

### TTS Response Times (GPU)
- **CSM TTS Generation**: ~0.5-1.0 seconds
- **OpenAI TTS Fallback**: ~1.0-2.0 seconds
- **Interrupt Latency**: <0.1 seconds
- **Model Preload Time**: ~3-5 seconds (one-time)

### Speech Recognition Performance
- **Hotword Detection**: <0.2 seconds
- **Speaker Identification**: ~0.3-0.5 seconds
- **Audio Chunk Reuse**: No additional delay
- **Unified STT Fallback**: ~0.5-1.0 seconds additional

### System Requirements
- **Minimum**: 8GB RAM, Intel i5 or equivalent
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA RTX 3080+ or better
- **Storage**: 3-7GB for models (cached locally)

## Documentation

### Complete Guides
- 📖 **[KYUTAI_STT_SETUP_GUIDE.md](KYUTAI_STT_SETUP_GUIDE.md)** - Complete Kyutai STT setup and configuration
- 📖 **[TTS_INTERRUPT_GUIDE.md](TTS_INTERRUPT_GUIDE.md)** - Complete TTS interrupt implementation guide
- 🔧 **[CSM_SETUP_GUIDE.md](CSM_SETUP_GUIDE.md)** - Sesame CSM setup and configuration
- ⚡ **[GPU_OPTIMIZATION_SUMMARY.md](GPU_OPTIMIZATION_SUMMARY.md)** - Performance optimization guide
- 📝 **[README.md](README.md)** - This comprehensive overview

### Quick Reference
- 🎙️ **Enhanced Assistant**: `python jarvis_enhanced.py` (dual STT with engine choice)
- 🎤 **Real-time Assistant**: `python jarvis.py` (RealTimeSTT with interrupts)
- 💬 **Text Assistant**: `python assist.py` (text-based with voice setup)
- 🧪 **Test Kyutai STT**: `python test_kyutai_stt.py` (verify STT setup)
- 🧪 **Test Interrupts**: `python test_interrupt.py` (verify interrupt functionality)
- 🔧 **Voice Setup**: `python assist.py --voice-settings` (voice management)

### API Reference
- **`unified_stt.transcribe_audio(audio_path)`** - Unified STT with automatic engine selection
- **`kyutai_stt.transcribe_with_kyutai(audio_path)`** - Direct Kyutai STT usage
- **`switch_stt_engine('kyutai'|'realtime')`** - Runtime engine switching
- **`assist.TTS_with_interrupt(text)`** - Interruptible TTS with status return
- **`assist.interrupt_tts()`** - Immediately stop active TTS playback
- **`assist.is_tts_active()`** - Check if TTS is currently playing
- **`voice_manager.identify_speaker(audio_path)`** - Speaker identification

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with proper documentation
4. Test thoroughly, especially TTS interrupt functionality
5. Run test suite: `python test_interrupt.py`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Submit a pull request

### Development Guidelines
- Follow existing code style and patterns
- Add comprehensive docstrings for new functions
- Test interrupt functionality thoroughly
- Update documentation for new features
- Maintain backward compatibility where possible

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support & Contact

- 🐛 **Issues**: Report bugs via GitHub Issues
- 💡 **Feature Requests**: Submit enhancement ideas
- 📧 **Contact**: Reach out via GitHub or email
- 📚 **Documentation**: Check the complete guides in the repository

## Acknowledgments

- **Hugging Face**: For the Sesame CSM TTS model
- **OpenAI**: For GPT models and TTS API
- **RealtimeSTT**: For real-time speech recognition
- **Community**: For feedback and contributions

---

**Bontle v3.0.0** - Advanced AI Assistant with Revolutionary Interrupt TTS System

*Experience natural, interruptible conversations with your AI assistant!* 🚀
