"""
Voice Signature Recognition Module for Bontle AI Assistant

This module handles voice signature registration, identification, and management.
It provides functionality to:
- Register speaker voice signatures
- Identify speakers from voice samples
- Manage voice database
- Extract voice features for speaker recognition
"""

import os
import json
import time
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine
import pyaudio
import wave
from datetime import datetime


class VoiceSignatureManager:
    def __init__(self, db_path="voice_signatures.json", recordings_path="voice_recordings"):
        self.db_path = db_path
        self.recordings_path = recordings_path
        self.voice_db = {}
        self.speaker_embeddings = {}
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(recordings_path):
            os.makedirs(recordings_path)
            
        # Load existing voice signatures
        self.load_voice_database()
        
        # Audio recording parameters
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
    def load_voice_database(self):
        """Load voice signatures from database file"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.voice_db = data.get('voice_db', {})
                    loaded_embeddings = data.get('speaker_embeddings', {})
                    
                    # Convert loaded embeddings back to numpy arrays
                    self.speaker_embeddings = {}
                    for speaker, embedding in loaded_embeddings.items():
                        self.speaker_embeddings[speaker] = np.array(embedding)
        except Exception as e:
            print(f"Error loading voice database: {e}")
            self.voice_db = {}
            self.speaker_embeddings = {}
    
    def update_all_features(self):
        """Re-extract features for all registered speakers (useful when algorithm changes)"""
        print("Updating features for all registered speakers...")
        updated = 0
        for speaker_name, speaker_info in self.voice_db.items():
            audio_path = speaker_info.get('audio_path')
            if audio_path and os.path.exists(audio_path):
                print(f"Updating features for {speaker_name}...")
                features = self.extract_voice_features(audio_path)
                if features is not None:
                    self.speaker_embeddings[speaker_name] = features
                    self.voice_db[speaker_name]['feature_vector_length'] = len(features)
                    updated += 1
                else:
                    print(f"Failed to update features for {speaker_name}")
            else:
                print(f"Audio file not found for {speaker_name}: {audio_path}")
        
        if updated > 0:
            self.save_voice_database()
            print(f"Successfully updated features for {updated} speakers")
        else:
            print("No speakers were updated")
    
    def save_voice_database(self):
        """Save voice signatures to database file"""
        try:
            data = {
                'voice_db': self.voice_db,
                'speaker_embeddings': self.speaker_embeddings
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        except Exception as e:
            print(f"Error saving voice database: {e}")
    
    def record_audio(self, duration=5, filename=None):
        """Record audio for voice signature registration or recognition"""
        if filename is None:
            filename = f"temp_recording_{int(time.time())}.wav"
        
        filepath = os.path.join(self.recordings_path, filename)
        
        print(f"Recording for {duration} seconds...")
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # Open stream
        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        
        # Record audio
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save the recording
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        print(f"Recording saved to: {filepath}")
        return filepath
    
    def extract_voice_features(self, audio_path):
        """Extract voice features from audio file using improved spectral and prosodic features"""
        try:
            # Load audio file
            audio_data, sr = sf.read(audio_path)
            
            # If stereo, convert to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to target sample rate if needed
            if sr != self.sample_rate:
                from scipy.signal import resample
                target_length = int(len(audio_data) * self.sample_rate / sr)
                audio_data = resample(audio_data, target_length)
            
            # Ensure minimum length for analysis
            min_length = self.sample_rate  # 1 second minimum
            if len(audio_data) < min_length:
                # Pad with zeros if too short
                audio_data = np.pad(audio_data, (0, min_length - len(audio_data)))
            
            # Improved feature extraction with fixed-size output
            
            # 1. Spectral features
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft)
            magnitude = magnitude[:len(magnitude)//2]  # Positive frequencies only
            
            # Spectral centroid - brightness of sound
            freqs = np.arange(len(magnitude))
            spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
            
            # Spectral rolloff - frequency below which 85% of energy is contained
            cumsum_magnitude = np.cumsum(magnitude)
            rolloff_threshold = 0.85 * cumsum_magnitude[-1]
            spectral_rolloff = np.where(cumsum_magnitude >= rolloff_threshold)[0]
            spectral_rolloff = spectral_rolloff[0] if len(spectral_rolloff) > 0 else len(magnitude) - 1
            
            # Spectral flux - measure of how quickly the power spectrum changes
            spectral_flux = np.mean(np.diff(magnitude)**2)
            
            # Zero crossing rate - measure of speech/noise characteristics
            zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
            zcr = len(zero_crossings) / len(audio_data)
            
            # 2. Energy features
            energy = np.sum(audio_data**2)
            rms_energy = np.sqrt(np.mean(audio_data**2))
            
            # 3. Pitch-related features (simplified)
            # Find peaks in autocorrelation to estimate fundamental frequency
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first peak after initial lag
            peaks = []
            for i in range(20, min(400, len(autocorr)-1)):  # Look for pitch in reasonable range
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append((i, autocorr[i]))
            
            fundamental_freq = 0
            if peaks:
                # Take the peak with highest correlation
                peaks.sort(key=lambda x: x[1], reverse=True)
                fundamental_freq = self.sample_rate / peaks[0][0] if peaks[0][0] > 0 else 0
            
            # 4. MFCC-like features (fixed size)
            # Ensure we always get 40 mel features by interpolating if necessary
            target_mel_size = 40
            if len(magnitude) >= target_mel_size:
                mel_features = np.log(magnitude[:target_mel_size] + 1e-8)
            else:
                # Interpolate to get target size
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(magnitude))
                x_new = np.linspace(0, 1, target_mel_size)
                f = interp1d(x_old, magnitude, kind='linear', fill_value="extrapolate")
                interpolated_magnitude = f(x_new)
                mel_features = np.log(interpolated_magnitude + 1e-8)
            
            # 5. Temporal features
            # Calculate short-term energy variation
            frame_length = 256
            frames = []
            for i in range(0, len(audio_data) - frame_length, frame_length // 2):
                frame = audio_data[i:i + frame_length]
                frames.append(np.sum(frame**2))
            
            energy_variance = np.var(frames) if len(frames) > 1 else 0
            
            # 6. Formant-like features (fixed size)
            formant_features = []
            num_formants = 3
            for i in range(num_formants):  # Extract 3 formant-like features
                start_freq = i * 1000  # Rough formant locations
                end_freq = (i + 1) * 1000
                start_bin = int(start_freq * len(magnitude) / (self.sample_rate / 2))
                end_bin = int(end_freq * len(magnitude) / (self.sample_rate / 2))
                if end_bin > len(magnitude):
                    end_bin = len(magnitude)
                formant_energy = np.mean(magnitude[start_bin:end_bin]) if end_bin > start_bin else 0
                formant_features.append(formant_energy)
            
            # Combine all features with fixed sizes
            basic_features = np.array([spectral_centroid, spectral_rolloff, spectral_flux, zcr,
                                     energy, rms_energy, fundamental_freq, energy_variance])
            
            # Ensure mel_features is exactly the right size
            mel_features = np.array(mel_features[:target_mel_size])
            if len(mel_features) < target_mel_size:
                mel_features = np.pad(mel_features, (0, target_mel_size - len(mel_features)))
            
            # Ensure formant_features is exactly the right size
            formant_features = np.array(formant_features[:num_formants])
            if len(formant_features) < num_formants:
                formant_features = np.pad(formant_features, (0, num_formants - len(formant_features)))
            
            # Combine all features
            features = np.concatenate([basic_features, mel_features, formant_features])
            
            # Normalize features to prevent any single feature from dominating
            features = features / (np.linalg.norm(features) + 1e-8)
            
            # Final check - ensure consistent size
            expected_size = 8 + target_mel_size + num_formants  # 8 + 40 + 3 = 51
            if len(features) != expected_size:
                # Force to expected size
                if len(features) > expected_size:
                    features = features[:expected_size]
                else:
                    features = np.pad(features, (0, expected_size - len(features)))
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def register_speaker(self, speaker_name, audio_path=None, duration=5):
        """Register a new speaker's voice signature"""
        try:
            # Record audio if path not provided
            if audio_path is None:
                print(f"Recording voice sample for {speaker_name}...")
                print("Please say something for voice registration (speak naturally for better recognition)")
                audio_path = self.record_audio(duration, f"{speaker_name}_{int(time.time())}.wav")
            
            # Extract voice features
            features = self.extract_voice_features(audio_path)
            
            if features is not None:
                # Store the voice signature
                self.speaker_embeddings[speaker_name] = features
                self.voice_db[speaker_name] = {
                    'audio_path': audio_path,
                    'registration_time': datetime.now().isoformat(),
                    'feature_vector_length': len(features)
                }
                
                # Save to database
                self.save_voice_database()
                
                print(f"Successfully registered voice signature for {speaker_name}")
                return True
            else:
                print(f"Failed to extract features for {speaker_name}")
                return False
                
        except Exception as e:
            print(f"Error registering speaker {speaker_name}: {e}")
            return False
    
    def identify_speaker(self, audio_path=None, duration=3, threshold=0.7):
        """Identify speaker from voice sample"""
        audio_provided_externally = audio_path is not None
        
        try:
            # Record audio if path not provided
            if audio_path is None:
                print("Recording voice sample for identification...")
                print("Please speak...")
                audio_path = self.record_audio(duration, f"identification_{int(time.time())}.wav")
            
            # Extract features from the audio
            features = self.extract_voice_features(audio_path)
            
            if features is None:
                return "Unknown", 0.0
            
            if not self.speaker_embeddings:
                return "Unknown - No registered speakers", 0.0
            
            best_match = None
            best_similarity = 0.0
            
            # Compare with all registered speakers
            for speaker_name, stored_features in self.speaker_embeddings.items():
                # Calculate cosine similarity
                similarity = 1 - cosine(features, stored_features)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_name
            
            # Clean up temporary file only if we created it internally
            if not audio_provided_externally and ("temp_recording" in audio_path or "identification" in audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            # Check if similarity meets threshold
            if best_similarity >= threshold:
                return best_match, best_similarity
            else:
                return "Unknown", best_similarity
                
        except Exception as e:
            print(f"Error identifying speaker: {e}")
            return "Unknown", 0.0
    
    def list_registered_speakers(self):
        """List all registered speakers"""
        return list(self.voice_db.keys())
    
    def remove_speaker(self, speaker_name):
        """Remove a speaker from the database"""
        try:
            if speaker_name in self.voice_db:
                # Remove audio file if it exists
                audio_path = self.voice_db[speaker_name].get('audio_path')
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
                
                # Remove from database
                del self.voice_db[speaker_name]
                del self.speaker_embeddings[speaker_name]
                
                # Save updated database
                self.save_voice_database()
                
                print(f"Successfully removed {speaker_name}")
                return True
            else:
                print(f"Speaker {speaker_name} not found")
                return False
        except Exception as e:
            print(f"Error removing speaker {speaker_name}: {e}")
            return False


# Global voice manager instance - initialized when module is imported
voice_manager = VoiceSignatureManager()


# Voice signature utility functions
def register_new_speaker(speaker_name, duration=5):
    """Register a new speaker's voice signature"""
    return voice_manager.register_speaker(speaker_name, duration=duration)


def identify_current_speaker(duration=3):
    """Identify the current speaker"""
    return voice_manager.identify_speaker(duration=duration)


def list_all_speakers():
    """List all registered speakers"""
    return voice_manager.list_registered_speakers()


def remove_registered_speaker(speaker_name):
    """Remove a speaker from the voice database"""
    return voice_manager.remove_speaker(speaker_name)


def get_speaker_info():
    """Get information about registered speakers"""
    speakers = voice_manager.list_registered_speakers()
    info = f"Registered speakers ({len(speakers)}): {', '.join(speakers) if speakers else 'None'}"
    return info


def quick_voice_setup():
    """Quick setup for voice recognition without full interface"""
    print("\n🎤 Quick Voice Setup")
    print("This will register your voice for speaker identification")
    
    name = input("👤 Enter your name: ").strip()
    if name:
        print(f"🔴 Recording voice sample for {name}...")
        print("📢 Please speak naturally for 5 seconds...")
        print("💡 Say something like: 'Hello, this is [your name], I'm setting up voice recognition for the Bontle assistant.'")
        
        time.sleep(2)  # Give user time to prepare
        success = register_new_speaker(name, duration=5)
        
        if success:
            print(f"✅ Voice registered successfully for {name}!")
            print("🎯 Voice identification will now work automatically when you use the assistant")
            return True
        else:
            print("❌ Voice registration failed. You can try again later.")
            return False
    else:
        print("❌ Please enter a valid name")
        return False


def test_voice_signature():
    """Test the voice signature recognition functionality"""
    print("=== Voice Signature Recognition Test ===")
    print(get_speaker_info())
    
    while True:
        print("\nVoice Signature Options:")
        print("1. Register new speaker")
        print("2. Identify speaker")
        print("3. List registered speakers")
        print("4. Remove speaker")
        print("5. Update all speaker features")
        print("6. Demo conversation with speaker identification")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == "1":
            name = input("Enter speaker name: ").strip()
            if name:
                print(f"Registering speaker: {name}")
                print("Get ready to speak...")
                time.sleep(2)
                success = register_new_speaker(name, duration=5)
                if success:
                    print(f"✓ Successfully registered {name}")
                else:
                    print(f"✗ Failed to register {name}")
            else:
                print("Please enter a valid name")
                
        elif choice == "2":
            print("Identifying speaker...")
            print("Get ready to speak...")
            time.sleep(2)
            speaker, confidence = identify_current_speaker(duration=3)
            print(f"Identified speaker: {speaker} (confidence: {confidence:.2f})")
            
        elif choice == "3":
            print(get_speaker_info())
            
        elif choice == "4":
            speakers = voice_manager.list_registered_speakers()
            if speakers:
                print("Registered speakers:")
                for i, speaker in enumerate(speakers, 1):
                    print(f"{i}. {speaker}")
                try:
                    choice_num = int(input("Enter speaker number to remove: "))
                    if 1 <= choice_num <= len(speakers):
                        speaker_to_remove = speakers[choice_num - 1]
                        success = remove_registered_speaker(speaker_to_remove)
                        if success:
                            print(f"✓ Successfully removed {speaker_to_remove}")
                        else:
                            print(f"✗ Failed to remove {speaker_to_remove}")
                    else:
                        print("Invalid speaker number")
                except ValueError:
                    print("Please enter a valid number")
            else:
                print("No speakers registered")
        
        elif choice == "5":
            voice_manager.update_all_features()
        
        elif choice == "6":
            demo_conversation_with_speaker_id()
                
        elif choice == "7":
            print("Exiting voice signature test")
            break
            
        else:
            print("Invalid choice. Please enter 1-7.")


def demo_conversation_with_speaker_id():
    """Demo showing conversation with automatic speaker identification"""
    print("\n=== Conversation Demo with Speaker Identification ===")
    print("This demo shows how the assistant can identify who is speaking")
    print("and maintain context in conversations.")
    print("\nTo test this properly, you would need:")
    print("1. Multiple people registered in the voice database")
    print("2. A speech-to-text system (like the RealtimeSTT mentioned in requirements)")
    print("3. Integration with the actual conversation flow")
    
    # Import ask_question_memory from assist module if available
    try:
        from assist import ask_question_memory
    except ImportError:
        print("⚠️ Cannot import ask_question_memory from assist.py")
        print("This demo requires integration with the main assistant functionality")
        return
    
    # Simulate a conversation scenario
    conversation_examples = [
        "Hello, how are you today?",
        "What's the weather like?", 
        "Can you help me with my schedule?",
        "Thank you for your help!"
    ]
    
    print("\nSimulated conversation with speaker identification:")
    for i, example_text in enumerate(conversation_examples, 1):
        print(f"\n--- Turn {i} ---")
        print("🎤 Press Enter when ready to speak, or 's' to simulate...")
        
        user_input = input().strip().lower()
        if user_input == 's':
            # Simulate speaker identification
            speakers = voice_manager.list_registered_speakers()
            if speakers:
                import random
                simulated_speaker = random.choice(speakers)
                simulated_confidence = random.uniform(0.7, 1.0)
                print(f"[SIMULATED] Identified speaker: {simulated_speaker} (confidence: {simulated_confidence:.2f})")
                
                # Simulate the conversation response
                response = ask_question_memory(example_text, identify_speaker=False)
                # Manually add speaker info for simulation
                response_with_speaker = f"[Speaker: {simulated_speaker}] {response}"
                print(f"Assistant: {response_with_speaker}")
            else:
                print("No speakers registered. Please register some speakers first.")
                break
        else:
            # Real speaker identification
            print("Identifying speaker from voice...")
            speaker, confidence = identify_current_speaker(duration=3)
            
            if speaker != "Unknown":
                print(f"Identified speaker: {speaker} (confidence: {confidence:.2f})")
                
                # Get the user's question via text input (in real implementation, this would be STT)
                question = input("What would you like to ask? (or use example): ").strip()
                if not question:
                    question = example_text
                
                # Process with speaker identification
                response = ask_question_memory(question, identify_speaker=False)
                print(f"Assistant: {response}")
            else:
                print(f"Speaker not recognized (confidence: {confidence:.2f})")
                print("You might want to register your voice first.")
                break
    
    print("\n=== Demo Complete ===")
    print("In a real implementation, this would:")
    print("- Automatically detect voice activity")
    print("- Convert speech to text using RealtimeSTT")
    print("- Identify the speaker")
    print("- Provide personalized responses")
    print("- Remember conversation context per speaker")


if __name__ == "__main__":
    """Run voice signature management interface when called directly"""
    print("🎤 Voice Signature Management System")
    print("This module provides voice recognition capabilities for the Bontle AI Assistant")
    test_voice_signature()