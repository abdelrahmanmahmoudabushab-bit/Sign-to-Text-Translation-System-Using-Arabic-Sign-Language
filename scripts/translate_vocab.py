"""
Script to automatically translate all 502 KArSL vocabulary terms from English to Arabic.
Preserves existing curated translations from app.DataLoader.

Usage:
    python translate_vocab.py
"""

import os
import sys
import csv
import re
import json

# Add project root to python search path (script is now in scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.DataLoader import arabic_labels as existing_labels, actions as existing_actions

def translate_term(translator, text):
    """Translate English term to Arabic, handling numeric and special cases."""
    text_clean = text.strip()
    
    # 1. If it's a number, return as is
    if text_clean.isdigit():
        return text_clean
        
    # 2. Check if we already have it in existing labels
    for idx, arabic in existing_labels.items():
        # Check matches in the existing list
        pass

    try:
        translated = translator.translate(text_clean)
        # Google Translate sometimes adds extra chars or keeps English, clean it up
        return translated.strip()
    except Exception as e:
        print(f"  Warning: Translation failed for '{text}': {e}")
        return text_clean

def main():
    # Reconfigure stdout to support printing Arabic characters on Windows console
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("KArSL Vocabulary Arabic Auto-Translator")
    print("=" * 60)
    
    # Install deep_translator if not imported
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("Error: deep_translator library not installed.")
        sys.exit(1)
        
    translator = GoogleTranslator(source='en', target='ar')
    
    csv_path = r"D:\signo v6\datasets\karsl-landmarks\train.csv"
    if not os.path.exists(csv_path):
        print(f"Error: train.csv not found at {csv_path}")
        sys.exit(1)
        
    # 1. Read all 502 mappings from CSV
    print(f"Reading sign mappings from {csv_path}...")
    csv_mappings = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_mappings[row['sign_id']] = row['sign']
            
    print(f"Found {len(csv_mappings)} unique signs in CSV.")
    
    # 2. Start building our final mapped dictionaries
    # We want contiguous indices [0, 501] for the 502 classes
    new_actions = {}
    new_arabic_labels = {}
    
    # Order signs by sign_id (0001 to 0502)
    all_sign_ids = sorted(csv_mappings.keys())
    
    # Map from sign_id to contiguous index [0, 501]
    # To keep compatibility, we map 0001 -> index 0, 0002 -> index 1, etc.
    for idx, sign_id in enumerate(all_sign_ids):
        new_actions[sign_id] = idx
        
    # 3. Translate labels
    print("Translating labels...")
    
    # Re-use existing hand-curated translations where possible
    # We map KArSL folder IDs (existing_actions keys) to their curated Arabic words (existing_labels values)
    folder_to_curated_arabic = {}
    for folder, label_idx in existing_actions.items():
        if label_idx in existing_labels:
            folder_to_curated_arabic[folder] = existing_labels[label_idx]
            
    for sign_id in all_sign_ids:
        idx = new_actions[sign_id]
        english_label = csv_mappings[sign_id]
        
        # Check if we have a hand-curated translation
        if sign_id in folder_to_curated_arabic:
            arabic_word = folder_to_curated_arabic[sign_id]
            print(f"  [{idx}] {sign_id} -> {english_label} -> (Curated) {arabic_word}")
        else:
            # Otherwise use translator
            arabic_word = translate_term(translator, english_label)
            print(f"  [{idx}] {sign_id} -> {english_label} -> (Auto) {arabic_word}")
            
        new_arabic_labels[idx] = arabic_word
        
    # 4. Write to label_config.json (the format DataLoader.py reads)
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app", "label_config.json"
    )

    print(f"\nWriting updated label config to {config_path}...")

    config_data = {
        "arabic_labels": {str(k): v for k, v in new_arabic_labels.items()},
        "actions": new_actions
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    print(f"label_config.json updated successfully with all {len(new_arabic_labels)} sign words!")

if __name__ == "__main__":
    main()
