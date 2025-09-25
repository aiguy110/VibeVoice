#!/usr/bin/env python3
"""
Check what VibeVoice models are available on Hugging Face.
"""

from huggingface_hub import list_models, model_info
from huggingface_hub.errors import RepositoryNotFoundError
import requests

def check_vibevoice_models():
    print("🔍 Checking available VibeVoice models...")

    # Known model IDs from the README
    known_models = [
        "microsoft/VibeVoice-1.5B",
        "microsoft/VibeVoice-Large"
    ]

    for model_id in known_models:
        print(f"\n📦 Checking {model_id}:")

        try:
            # Try to get model info without authentication
            info = model_info(model_id)
            print(f"  ✅ Status: Available")
            print(f"  🔒 Gated: {'Yes' if info.gated else 'No'}")
            print(f"  📊 Downloads: {info.downloads if hasattr(info, 'downloads') else 'N/A'}")

        except RepositoryNotFoundError:
            print(f"  ❌ Status: Not found or requires authentication")

        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                print(f"  🔐 Status: Requires authentication (gated model)")
            else:
                print(f"  ❓ Status: Error - {e}")

    # Also search for any VibeVoice models
    print(f"\n🔎 Searching for all VibeVoice models...")
    try:
        models = list(list_models(search="VibeVoice", limit=20))
        if models:
            print(f"Found {len(models)} models:")
            for model in models:
                print(f"  - {model.id}")
        else:
            print("  No models found in search")
    except Exception as e:
        print(f"  Error searching: {e}")

if __name__ == "__main__":
    check_vibevoice_models()
