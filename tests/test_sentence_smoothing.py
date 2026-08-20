#!/usr/bin/env python3
import os
import sys

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm_util import smooth_sign_sentence

def test_sentence_reconstruction():
    print("="*65)
    print("TESTING JORDANIAN SIGN LANGUAGE SENTENCE RECONSTRUCTION & SMOOTHING")
    print("="*65)
    
    # Test case 1: Sequence of isolated sign glosses
    raw_signs = ["أنا", "يريد", "ذهاب", "مستشفى", "اليوم"]
    dialect = "Jordanian Arabic Sign Language"
    
    print(f"\n1. Raw Input Sign Sequence: {raw_signs}")
    print(f"Target Dialect:           {dialect}")
    
    result = smooth_sign_sentence(raw_signs, dialect=dialect)
    
    print("\n--- RECONSTRUCTED OUTPUT ---")
    print(f"Arabic Sentence (Jordanian Colloquial): {result.get('arabic')}")
    print(f"English Translation:                    {result.get('english')}")
    print("="*65)

if __name__ == "__main__":
    test_sentence_reconstruction()
