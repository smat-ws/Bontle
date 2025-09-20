#!/usr/bin/env python3
"""
Test script for TTS interrupt functionality
"""

import assist
import threading
import time

def test_interrupt_after_delay(delay_seconds=3):
    """Test function that interrupts TTS after a delay"""
    def interrupt_after_delay():
        time.sleep(delay_seconds)
        print(f"\n🔄 Interrupting TTS after {delay_seconds} seconds...")
        assist.interrupt_tts()
    
    # Start interrupt thread
    interrupt_thread = threading.Thread(target=interrupt_after_delay)
    interrupt_thread.daemon = True
    interrupt_thread.start()

def main():
    print("🎙️ Testing TTS with Interrupt Functionality")
    print("=" * 50)
    
    # Test 1: Basic interrupt functionality
    print("\n📝 Test 1: Basic TTS with interrupt after 3 seconds")
    test_text = "This is a test of the text to speech system with interrupt capability. I will keep talking for a while to demonstrate the interrupt functionality working properly during playback."
    
    test_interrupt_after_delay(3)
    result = assist.TTS_with_interrupt(test_text)
    print(f"✅ TTS Result: {result}")
    
    # Test 2: Short text that finishes before interrupt
    print("\n📝 Test 2: Short text that completes normally")
    short_text = "Short text test."
    result = assist.TTS_with_interrupt(short_text)
    print(f"✅ TTS Result: {result}")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()