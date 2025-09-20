#!/usr/bin/env python3
"""
Robust Kyutai STT Model Downloader
Downloads the Kyutai STT-2.6B-EN model with proper timeout handling and retries
"""

import os
import time
from huggingface_hub import hf_hub_download, login
from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration
import requests
from pathlib import Path

def setup_auth():
    """Setup Hugging Face authentication"""
    hf_token = "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"
    if hf_token:
        login(token=hf_token, add_to_git_credential=True)
        print("🔐 Authenticated with Hugging Face")
        return True
    else:
        print("❌ No valid Hugging Face token found")
        return False

def download_with_retry(repo_id, filename, max_retries=3):
    """Download a file with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading {filename} (attempt {attempt + 1}/{max_retries})")
            
            # Download with resume capability
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                resume_download=True  # Enable resume functionality
            )
            
            print(f"✅ Successfully downloaded {filename}")
            return file_path
            
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            print(f"⏰ Timeout occurred for {filename} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # Exponential backoff
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed to download {filename} after {max_retries} attempts")
                raise
        except Exception as e:
            print(f"❌ Unexpected error downloading {filename}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed to download {filename} after {max_retries} attempts")
                raise

def main():
    """Main download function"""
    print("🚀 Starting Kyutai STT Model Download")
    
    # Setup authentication
    if not setup_auth():
        return False
    
    repo_id = "kyutai/stt-2.6b-en-trfs"
    
    # Files to download (start with largest first to fail fast if there are issues)
    files_to_download = [
        "model-00001-of-00002.safetensors",  # ~4.37 GB
        "model-00002-of-00002.safetensors",  # ~1.55 GB
        "config.json",
        "generation_config.json", 
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    print(f"📋 Will download {len(files_to_download)} files from {repo_id}")
    
    success_count = 0
    for i, filename in enumerate(files_to_download, 1):
        try:
            print(f"\n[{i}/{len(files_to_download)}] Processing {filename}")
            file_path = download_with_retry(repo_id, filename, max_retries=3)
            success_count += 1
            print(f"💾 File saved to: {file_path}")
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            continue
    
    print(f"\n📊 Download Summary:")
    print(f"✅ Successfully downloaded: {success_count}/{len(files_to_download)} files")
    
    if success_count >= 8:  # At least got the model files + most config files
        print("🎉 Download completed successfully!")
        
        # Try to load the model to verify it works
        try:
            print("\n🧪 Testing model loading...")
            processor = KyutaiSpeechToTextProcessor.from_pretrained(repo_id)
            print("✅ Processor loaded successfully!")
            
            model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
                repo_id,
                torch_dtype="auto",
                device_map="auto" if os.environ.get('STT_USE_GPU', 'True').lower() == 'true' else "cpu"
            )
            print("✅ Model loaded successfully!")
            print("🎯 Kyutai STT is ready to use!")
            return True
            
        except Exception as e:
            print(f"⚠️  Model files downloaded but loading failed: {e}")
            print("💡 You may need to restart Python to clear any cached imports")
            return False
    else:
        print("❌ Download incomplete. Please check your internet connection and try again.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)