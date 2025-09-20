# Sesame CSM TTS Integration Guide

## Overview
This guide explains how to set up and use the Sesame CSM (Conversational Speech Model) for local text-to-speech generation in your Bontle voice assistant.

## What is Sesame CSM?
- **High-quality TTS**: State-of-the-art speech synthesis model from Sesame
- **Local generation**: Runs entirely on your machine for privacy and speed
- **Multi-speaker support**: Generate different voice characteristics
- **Real-time capable**: Fast enough for conversational AI applications

## Prerequisites
- Python 3.8+ with PyTorch
- Hugging Face account with access to gated models
- GPU recommended (but CPU works)
- ~3GB free disk space for model weights

## Setup Instructions

### 1. Get Hugging Face Access
1. Create account at [huggingface.co](https://huggingface.co)
2. Visit [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)
3. **Accept the model conditions** (required for gated model)
4. Go to [Settings > Tokens](https://huggingface.co/settings/tokens)
5. Create a new token with "Read" permissions

### 2. Configure Environment
1. Open your `.env` file in the project root
2. Replace `your_hf_token_here` with your actual token:
   ```
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
3. Save the file

### 3. Install Dependencies
The required packages should already be installed, but if needed:
```bash
py -m pip install transformers>=4.52.1 accelerate huggingface_hub
```

### 4. Test the Installation
Run the CSM test script:
```bash
py csm_tts.py
```

If successful, you should see:
- Model loading confirmation
- Audio generation tests
- Audio files saved to disk

## Usage

### Basic Usage
```python
from csm_tts import csm_text_to_speech

# Simple text-to-speech
result = csm_text_to_speech("Hello, this is a test of CSM TTS!")
```

### Advanced Usage
```python
from csm_tts import SesameCSM

# Initialize TTS engine
tts = SesameCSM()

# Generate with specific speaker
tts.text_to_speech(
    "This is speaker 1 talking",
    speaker_id="1",
    temperature=0.8,
    save_file="output.wav"
)
```

### Integration with Bontle
The CSM TTS is already integrated into your Bontle assistant:
- Set `USE_LOCAL_CSM_TTS = True` in `assist.py` (already done)
- The assistant will automatically use CSM for speech generation
- Falls back to OpenAI TTS if CSM fails

## Configuration Options

### Speaker IDs
- `"0"` - Default voice (neutral)
- `"1"` - Alternative voice 1
- `"2"` - Alternative voice 2
- Custom IDs work too

### Temperature Settings
- `0.1-0.5` - More consistent, robotic
- `0.6-0.8` - Balanced, natural (recommended)
- `0.9-1.0` - More expressive, variable

### Audio Quality
- Sample rate: 24kHz (high quality)
- Format: WAV (uncompressed)
- Bit depth: 16-bit

## Performance Tips

### GPU Acceleration
- CUDA: Automatic detection and usage
- Apple Silicon: MPS backend support
- CPU: Works but slower (~5-10x)

### Memory Management
- Model uses ~2-3GB VRAM/RAM
- Audio generation: ~100MB per minute
- Batch processing supported for efficiency

### Speed Optimization
- First generation is slower (model loading)
- Subsequent calls are much faster
- Consider keeping model loaded for real-time apps

## Troubleshooting

### Authentication Errors
```
401 Client Error: Unauthorized
```
**Solution**: Check your HF token and model access permissions

### Import Errors
```
ImportError: No module named 'transformers'
```
**Solution**: Update transformers to 4.52.1+
```bash
py -m pip install transformers>=4.52.1
```

### Memory Errors
```
CUDA out of memory
```
**Solutions**:
- Use CPU: Set device to "cpu" in csm_tts.py
- Reduce batch size
- Close other GPU applications

### Audio Playback Issues
```
pygame mixer error
```
**Solutions**:
- Install audio drivers
- Check system audio settings
- Use save_file option instead of direct playback

## Model Details

### Architecture
- Base: Llama backbone with audio decoder
- Audio codec: Mimi (24kHz)
- Size: 1B parameters
- Training: Conversational speech data

### Capabilities
- Text-to-speech synthesis
- Multi-speaker voice generation
- Contextual speech (with conversation history)
- Real-time generation
- High naturalness and intelligibility

### Limitations
- English language primarily
- Requires internet for initial download
- Large model size
- GPU recommended for best performance

## Files Modified
- `csm_tts.py` - Main CSM implementation
- `assist.py` - Integration with Bontle assistant
- `requirements.txt` - Added CSM dependencies
- `.env` - Added HF token configuration

## Support
If you encounter issues:
1. Check the console output for specific error messages
2. Verify your HF token has the correct permissions
3. Ensure you've accepted the model conditions
4. Try the test script first: `py csm_tts.py`

## Next Steps
Once CSM is working:
- Experiment with different speaker IDs
- Tune temperature for your preferred voice style
- Consider fine-tuning for specific voices
- Explore conversation context features

---
*Sesame CSM integration completed successfully! 🎉*