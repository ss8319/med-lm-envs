#!/usr/bin/env python3
"""
Comprehensive token count analysis for LongHealth benchmark using GPT-4.1-mini tokenizer.
Analyzes the complete JSON structure including texts, questions, answers, and metadata.
"""

import json
import tiktoken
import os

def count_tokens():
    # Load the benchmark data
    benchmark_path = "benchmark_v5.json"
    if not os.path.exists(benchmark_path):
        print(f"Error: {benchmark_path} not found. Run this script from the longhealth directory.")
        return
    
    # Use GPT-4 tokenizer (GPT-4.1-mini uses same tokenizer)
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Initialize counters
    total_tokens = 0
    patient_stats = []
    
    # Breakdown counters
    texts_tokens = 0
    questions_tokens = 0
    answers_tokens = 0
    metadata_tokens = 0
    
    print("=== LongHealth Complete Token Count Analysis ===")
    print()
    
    for patient_id, patient_data in data.items():
        patient_total = 0
        
        # Count tokens in clinical texts
        patient_texts_tokens = 0
        for text_id, text_content in patient_data["texts"].items():
            text_tokens = len(tokenizer.encode(text_content))
            patient_texts_tokens += text_tokens
            texts_tokens += text_tokens
        
        # Count tokens in questions and answers
        patient_questions_tokens = 0
        patient_answers_tokens = 0
        
        for question in patient_data["questions"]:
            # Question text
            question_tokens = len(tokenizer.encode(question["question"]))
            patient_questions_tokens += question_tokens
            questions_tokens += question_tokens
            
            # Answer options (A, B, C, D, E)
            for option_key in ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"]:
                option_tokens = len(tokenizer.encode(question[option_key]))
                patient_answers_tokens += option_tokens
                answers_tokens += option_tokens
            
            # Correct answer
            correct_tokens = len(tokenizer.encode(question["correct"]))
            patient_answers_tokens += correct_tokens
            answers_tokens += correct_tokens
        
        # Count tokens in metadata (name, birthday, diagnosis)
        patient_metadata_tokens = 0
        metadata_fields = ["name", "birthday", "diagnosis"]
        for field in metadata_fields:
            if field in patient_data:
                field_tokens = len(tokenizer.encode(str(patient_data[field])))
                patient_metadata_tokens += field_tokens
                metadata_tokens += field_tokens
        
        # Total for this patient
        patient_total = patient_texts_tokens + patient_questions_tokens + patient_answers_tokens + patient_metadata_tokens
        total_tokens += patient_total
        
        patient_stats.append({
            "patient_id": patient_id,
            "total_tokens": patient_total,
            "texts_tokens": patient_texts_tokens,
            "questions_tokens": patient_questions_tokens,
            "answers_tokens": patient_answers_tokens,
            "metadata_tokens": patient_metadata_tokens
        })
    
    # Print summary
    print("SUMMARY STATISTICS")
    print(f"Total tokens in entire dataset: {total_tokens:,}")
    print(f"Total patients: {len(data)}")
    print(f"Total questions: {sum(len(patient['questions']) for patient in data.values())}")
    print()
    
    print("BREAKDOWN BY CONTENT TYPE")
    print(f"Clinical texts: {texts_tokens:,} tokens ({texts_tokens/total_tokens*100:.1f}%)")
    print(f"Questions: {questions_tokens:,} tokens ({questions_tokens/total_tokens*100:.1f}%)")
    print(f"Answer options: {answers_tokens:,} tokens ({answers_tokens/total_tokens*100:.1f}%)")
    print(f"Metadata: {metadata_tokens:,} tokens ({metadata_tokens/total_tokens*100:.1f}%)")
    print()
    
    print("AVERAGES")
    print(f"Average tokens per patient: {total_tokens/len(data):,.0f}")
    print(f"Average tokens per question: {total_tokens/400:,.0f}")
    print(f"Average tokens per clinical text: {texts_tokens/sum(len(patient['texts']) for patient in data.values()):,.0f}")
    print()
    
    # Show per-patient breakdown (top 10)
    print("PER-PATIENT BREAKDOWN (Top 10 by total tokens)")
    sorted_patients = sorted(patient_stats, key=lambda x: x["total_tokens"], reverse=True)
    for i, patient in enumerate(sorted_patients[:10]):
        print(f"{i+1:2d}. {patient['patient_id']}: {patient['total_tokens']:,} tokens")
        print(f"     Texts: {patient['texts_tokens']:,} | Questions: {patient['questions_tokens']:,} | Answers: {patient['answers_tokens']:,} | Metadata: {patient['metadata_tokens']:,}")
    
    if len(sorted_patients) > 10:
        print(f"     ... and {len(sorted_patients)-10} more patients")
    print()
    
    # Context window analysis
    print("CONTEXT WINDOW ANALYSIS")
    context_windows = [4000, 8000, 16000, 32000, 128000]
    for window in context_windows:
        patients_fit = sum(1 for p in patient_stats if p["texts_tokens"] <= window)
        percentage = patients_fit / len(data) * 100
        print(f"Patients fitting in {window:,} token context: {patients_fit}/{len(data)} ({percentage:.1f}%)")
    
    print()
    print("NOTES")
    print("- This counts ALL content in the JSON file (texts, questions, answers, metadata)")
    print("- Clinical texts are the main content (largest token count)")
    print("- Questions and answers are relatively small compared to clinical texts")
    print("- Use this for understanding total dataset size and context requirements")

if __name__ == "__main__":
    count_tokens()
