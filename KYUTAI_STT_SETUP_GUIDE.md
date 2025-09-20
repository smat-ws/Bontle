# Kyutai STT Integration Guide

## Overview
This guide explains how to set up and use the Kyutai STT-2.6B-EN model for high-quality speech-to-text in your Bontle voice assistant, with intelligent fallback to RealTimeSTT.

## What is Kyutai STT?
- **State-of-the-art STT**: Advanced speech-to-text model with 2.6B parameters
- **High Accuracy**: Superior transcription quality, especially for longer speech
- **GPU Accelerated**: Optimized for NVIDIA GPUs and Apple Silicon
- **Streaming Capable**: Supports real-time streaming transcription
- **English Optimized**: Specialized for English language transcription

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Unified STT     │───▶│  Bontle AI      │
│                 │    │    Manager       │    │   Assistant     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  STT Engine Choice  │
                    │                     │
                    │  ┌─────────────────┐│
                    │  │  Kyutai STT     ││ ◀─ Primary (Default)
                    │  │  (High Quality) ││
                    │  └─────────────────┘│
                    │           │         │
                    │           ▼         │
                    │  ┌─────────────────┐│
                    │  │  RealTimeSTT    ││ ◀─ Fallback
                    │  │  (Real-time)    ││
                    │  └─────────────────┘│
                    └─────────────────────┘
```

## Prerequisites

### System Requirements
- **Python 3.8+** with PyTorch support
- **GPU Recommended**: NVIDIA GPU with 4GB+ VRAM (or Apple Silicon)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~3GB for model weights
- **Internet**: Required for initial model download

### Dependencies
- `transformers >= 4.53.0` (Critical - Kyutai support added in this version)
- `torch` with CUDA support (for GPU acceleration)
- `numpy`, `scipy`, `soundfile` (audio processing)
- `pyaudio` (audio recording)
- `huggingface_hub` (model downloading)

## Setup Instructions

### 1. Update Dependencies

First, ensure you have the latest transformers version:

```bash
pip install transformers>=4.53.0
```

For GPU acceleration (highly recommended):
```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Hugging Face Access

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to [Settings > Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with "Read" permissions
4. Add to your `.env` file:
   ```env
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```

### 3. Configure STT Settings

The `.env` file includes comprehensive STT configuration:

```env
# STT Engine Selection
STT_ENGINE=kyutai              # Primary engine: kyutai or realtime
STT_USE_GPU=True              # Enable GPU acceleration
STT_FALLBACK_ENABLED=True     # Enable fallback to other engines

# Kyutai STT Settings
KYUTAI_MODEL=kyutai/stt-2.6b-en-trfs
KYUTAI_TEMPERATURE=0.8
KYUTAI_MAX_TOKENS=512

# RealTimeSTT Settings (fallback)
REALTIME_STT_MODEL=medium.en
REALTIME_STT_LANGUAGE=en
REALTIME_STT_SILENCE=0.15
REALTIME_STT_SENSITIVITY=0.4
```

### 4. Test the Installation

Run the comprehensive test suite:
```bash
python test_kyutai_stt.py
```

This will verify:
- ✅ All dependencies are installed
- ✅ Transformers version supports Kyutai STT
- ✅ GPU is available and configured
- ✅ Hugging Face token is valid
- ✅ Kyutai STT can be loaded
- ✅ Unified STT system works

## Usage

### Enhanced Assistant (Recommended)

Run the enhanced assistant with Kyutai STT support:
```bash
python jarvis_enhanced.py
```

**Features:**
- 🎯 **Engine Selection**: Choose between Kyutai STT and RealTimeSTT
- 🔄 **Intelligent Fallback**: Automatic fallback if primary engine fails
- ⚡ **GPU Acceleration**: Optimized for maximum performance
- 🎙️ **Multiple Modes**: Continuous listening or command-based recording

### Direct Kyutai STT Testing

Test Kyutai STT directly:
```bash
python kyutai_stt.py
```

### Configuration Management

```python
# Check STT status
from unified_stt import get_stt_status, get_available_stt_engines

status = get_stt_status()
print(f"Active: {status['active_engine']}")
print(f"Available: {get_available_stt_engines()}")

# Switch engines
from unified_stt import switch_stt_engine
switch_stt_engine('kyutai')  # or 'realtime'
```

## Performance Comparison

| Feature | Kyutai STT | RealTimeSTT |
|---------|------------|-------------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **GPU Usage** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Real-time** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Long Audio** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Delay** | 2.5 seconds | < 0.5 seconds |

### When to Use Each Engine

**Use Kyutai STT when:**
- ✅ Maximum transcription accuracy is needed
- ✅ Processing longer audio segments
- ✅ GPU is available for acceleration
- ✅ 2.5 second delay is acceptable

**Use RealTimeSTT when:**
- ✅ Real-time response is critical
- ✅ Interactive conversations with minimal delay
- ✅ Limited GPU resources
- ✅ Continuous listening mode

## Advanced Configuration

### GPU Optimization

For maximum performance with Kyutai STT:

1. **VRAM Requirements**:
   - Minimum: 4GB VRAM
   - Recommended: 8GB+ VRAM
   - Optimal: 12GB+ VRAM

2. **Optimization Settings**:
   ```python
   # In kyutai_stt.py - automatic GPU optimizations
   torch_dtype=torch.float16     # FP16 for 2x speed
   device_map="cuda:0"           # Force GPU placement
   low_cpu_mem_usage=True        # Optimize memory
   ```

3. **Batch Processing**:
   ```python
   from kyutai_stt import initialize_kyutai_stt
   stt = initialize_kyutai_stt()
   
   # Process multiple audio files efficiently
   texts = stt.batch_transcribe([audio1, audio2, audio3])
   ```

### Custom Model Configuration

```python
# Use different Kyutai models
from kyutai_stt import KyutaiSTT

# English + French model (smaller, faster)
stt_multilingual = KyutaiSTT("kyutai/stt-1b-en_fr")

# Custom settings
stt_custom = KyutaiSTT()
stt_custom.sample_rate = 24000  # Adjust as needed
```

### Integration with Voice Assistant

```python
# In your assistant code
from unified_stt import transcribe_audio

def process_voice_command():
    # Record audio (your implementation)
    audio_data = record_audio()
    
    # Transcribe with automatic engine selection
    text = transcribe_audio(audio_array=audio_data)
    
    # Process with assistant
    response = assistant.process(text)
    return response
```

## Troubleshooting

### Common Issues

#### 1. Transformers Version Error
```
ImportError: cannot import name 'KyutaiSpeechToTextProcessor'
```
**Solution**: Update transformers
```bash
pip install transformers>=4.53.0
```

#### 2. GPU Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Use smaller model: `kyutai/stt-1b-en_fr`
- Reduce batch size
- Enable CPU fallback: `STT_USE_GPU=False`
- Close other GPU applications

#### 3. Model Download Fails
```
ConnectionError or 401 Unauthorized
```
**Solutions**:
- Check internet connection
- Verify Hugging Face token in `.env`
- Try VPN if regional restrictions
- Use manual download if needed

#### 4. Audio Recording Issues
```
OSError: No Default Input Device Available
```
**Solutions**:
- Check microphone permissions
- Verify audio device availability
- Install PyAudio properly: `pip install pyaudio`

#### 5. Import Errors
```
ModuleNotFoundError: No module named 'kyutai_stt'
```
**Solution**: Ensure you're in the correct directory and all files are present

### Performance Optimization

#### For CPU-Only Systems
```env
STT_ENGINE=realtime
STT_USE_GPU=False
STT_FALLBACK_ENABLED=True
```

#### For GPU Systems
```env
STT_ENGINE=kyutai
STT_USE_GPU=True
KYUTAI_TEMPERATURE=0.7  # Lower for more consistent results
```

#### Memory Optimization
- Use FP16: Automatic on GPU
- Clear cache: Automatic after each transcription
- Limit audio length: Max 30 seconds recommended

## File Structure

After setup, your project will include:

```
Bontle/
├── kyutai_stt.py              # Kyutai STT implementation
├── stt_config.py              # STT configuration management
├── unified_stt.py             # Unified STT manager with fallback
├── jarvis_enhanced.py         # Enhanced assistant with STT choice
├── test_kyutai_stt.py         # Comprehensive test suite
├── requirements.txt           # Updated dependencies
├── .env                       # Configuration with STT settings
├── KYUTAI_STT_SETUP_GUIDE.md  # This guide
└── README.md                  # Updated project documentation
```

## Usage Examples

### Quick Start
```bash
# Test the system
python test_kyutai_stt.py

# Run enhanced assistant
python jarvis_enhanced.py

# Select Kyutai STT when prompted
# Say "Bontle" or "Jarvis" to activate
```

### Python API Usage
```python
# Direct Kyutai STT usage
from kyutai_stt import transcribe_with_kyutai

text = transcribe_with_kyutai(audio_path="recording.wav")
print(f"Transcription: {text}")

# Unified STT with fallback
from unified_stt import transcribe_audio

text = transcribe_audio(audio_path="recording.wav")
print(f"Transcription: {text}")
```

### Configuration Changes
```python
# Switch engines at runtime
from unified_stt import switch_stt_engine
switch_stt_engine('kyutai')  # High quality
switch_stt_engine('realtime')  # Low latency
```

## Next Steps

1. **Test Performance**: Compare Kyutai STT vs RealTimeSTT on your audio
2. **Optimize Settings**: Adjust temperature and token limits for your use case
3. **Integrate Features**: Combine with voice signature recognition
4. **Monitor Resources**: Check GPU/CPU usage during operation
5. **Scale Usage**: Consider batch processing for multiple audio files

## Support

If you encounter issues:

1. **Run Tests**: `python test_kyutai_stt.py`
2. **Check Logs**: Look for specific error messages
3. **Verify Setup**: Ensure all dependencies are correct versions
4. **Test Fallback**: Try switching to RealTimeSTT
5. **Check Resources**: Monitor GPU/RAM usage

---

**Kyutai STT Integration Complete! 🎉**

You now have access to state-of-the-art speech recognition with intelligent fallback capabilities.

*Default: Kyutai STT with RealTimeSTT fallback for optimal performance and reliability.*