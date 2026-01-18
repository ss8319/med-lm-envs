#!/usr/bin/env python3
"""
Quick script to estimate token counts in LongHealth benchmark.
Uses character-based estimation since tiktoken installation is having issues.
"""

import json
import os

def count_tokens():
    # Load the benchmark data
    benchmark_path = "environments/longhealth/benchmark_v5.json"
    if not os.path.exists(benchmark_path):
        print(f"Error: {benchmark_path} not found. Run this script from the project root.")
        return
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Count characters for all clinical texts
    total_chars = 0
    patient_stats = []
    
    for patient_id, patient_data in data.items():
        patient_chars = sum(len(text) for text in patient_data["texts"].values())
        total_chars += patient_chars
        patient_stats.append((patient_id, patient_chars))
    
    # Estimate tokens using different ratios
    print("=== LongHealth Token Count Analysis ===")
    print(f"Total characters: {total_chars:,}")
    print(f"Total patients: {len(data)}")
    print(f"Total questions: 400")
    print()
    
    print("Token estimates (different tokenizers):")
    print(f"  GPT-4 style (0.25 chars/token): {int(total_chars*0.25):,} tokens")
    print(f"  LLaMA style (0.33 chars/token): {int(total_chars*0.33):,} tokens")
    print(f"  Conservative (0.4 chars/token): {int(total_chars*0.4):,} tokens")
    print()
    
    print("Per-patient character counts:")
    for patient_id, chars in sorted(patient_stats, key=lambda x: x[1], reverse=True):
        print(f"  {patient_id}: {chars:,} chars")
    
    print()
    print("Note: These are estimates. For exact GPT-4.1-mini token counts, install tiktoken:")
    print("  pip install tiktoken")
    print("  Then run: python -c \"import tiktoken; tokenizer=tiktoken.encoding_for_model('gpt-4'); print('GPT-4 tokens:', len(tokenizer.encode('your text here')))\"")

if __name__ == "__main__":
    count_tokens()

