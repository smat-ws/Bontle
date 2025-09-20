# TTS with Interrupt Implementation Guide

## Overview

The TTS (Text-to-Speech) system has been enhanced with interrupt functionality, allowing seamless interruption of speech playback when new hotwords are detected. This creates a more natural, responsive interaction experience.

## Key Features

### 🔄 Interrupt Capability
- TTS playback can be interrupted at any time by detecting hotwords
- Smooth transition from interrupted TTS to processing new commands
- Proper cleanup of audio resources during interruption

### ⚡ Seamless Integration
- Uses the same audio from hotword detection for speaker identification
- No additional recording delays
- Maintains conversation context

### 🎙️ Dual TTS Support
- Primary: Sesame CSM TTS (local, GPU-accelerated)
- Fallback: OpenAI TTS (when CSM fails)
- Both support interrupt functionality

## Implementation Details

### New Functions in `assist.py`

#### `TTS_with_interrupt(text)`
- Replaces the standard `TTS()` function for interruptible playback
- Generates and plays audio with continuous interrupt monitoring
- Returns "done", "interrupted", or "error"

#### `interrupt_tts()`
- Signals active TTS to stop immediately
- Called when hotwords are detected during playback
- Thread-safe interrupt mechanism

#### `is_tts_active()`
- Checks if TTS is currently playing
- Used by jarvis.py to determine when interrupts are relevant

### Modified `jarvis.py` Logic

```python
# Check for hotword during TTS playback
if assist.is_tts_active() and any(hot_word in current_text.lower() for hot_word in hot_words):
    print("🔄 Hotword detected during TTS - interrupting...")
    assist.interrupt_tts()

# Use interruptible TTS
done = assist.TTS_with_interrupt(speech)
```

## Usage Examples

### Basic Usage
```python
import assist

# Standard interruptible TTS
result = assist.TTS_with_interrupt("Hello, this is a test message.")
if result == "interrupted":
    print("TTS was interrupted by user")
elif result == "done":
    print("TTS completed successfully")
```

### Interrupt Control
```python
import assist
import threading
import time

# Start TTS in background
def play_long_message():
    assist.TTS_with_interrupt("This is a very long message that can be interrupted...")

tts_thread = threading.Thread(target=play_long_message)
tts_thread.start()

# Interrupt after 2 seconds
time.sleep(2)
assist.interrupt_tts()
```

## Thread Safety

The interrupt system uses threading events for safe communication:
- `tts_interrupt_flag`: Signals when to interrupt
- `tts_playback_active`: Tracks if TTS is currently playing

## File Management

### CSM TTS
- Generates audio to `output.wav`
- Automatically cleans up file after playback or interruption

### OpenAI TTS
- Generates temporary `speech.mp3` files
- Proper cleanup on completion, interruption, or error

## Error Handling

### Fallback Chain
1. Try CSM TTS with interrupt support
2. If CSM fails, fall back to OpenAI TTS with interrupt support
3. Proper error reporting and resource cleanup

### Interrupt During Generation
- If interrupted during audio generation, generation stops
- Files are cleaned up appropriately
- Returns "interrupted" status immediately

## Integration with RealtimeSTT

### Seamless Speaker ID
- Uses captured hotword audio for speaker identification
- No additional recording delay
- Maintains natural conversation flow

### Hotword Detection
- Continuously monitors for hotwords even during TTS playback
- Immediate interrupt when hotwords detected
- Smooth transition to new command processing

## Configuration

### In `assist.py`
```python
USE_LOCAL_CSM_TTS = True  # Set to False to use OpenAI TTS as primary
```

### Hotword List (in `jarvis.py`)
```python
hot_words = ["bontle", "jarvis", "hi"]
```

## Testing

Run the test script to verify interrupt functionality:
```bash
python test_interrupt.py
```

This will:
1. Test interrupt after 3 seconds on long text
2. Test normal completion on short text
3. Verify proper cleanup and status reporting

## Benefits

### User Experience
- ✅ Natural interruption like human conversation
- ✅ No waiting for TTS to finish before giving new commands
- ✅ Immediate response to hotwords
- ✅ Smooth conversation flow

### Performance
- ✅ Minimal latency for interruption
- ✅ Proper resource cleanup
- ✅ Thread-safe implementation
- ✅ GPU acceleration maintained for CSM TTS

### Reliability
- ✅ Fallback support for TTS failures
- ✅ Error handling during interruption
- ✅ File cleanup on all exit paths
- ✅ Graceful degradation

## Troubleshooting

### Common Issues

1. **TTS not interrupting**: Check that `jarvis.py` is calling `interrupt_tts()` when hotwords are detected
2. **File not found errors**: Ensure proper cleanup in both CSM and OpenAI TTS paths
3. **Audio playback issues**: Verify pygame mixer initialization and audio file generation

### Debug Mode
Add debug prints to track interrupt flow:
```python
print(f"TTS active: {assist.is_tts_active()}")
print(f"Interrupt flag set: {assist.tts_interrupt_flag.is_set()}")
```

This implementation provides a robust, natural, and responsive TTS system that enhances the user experience significantly.