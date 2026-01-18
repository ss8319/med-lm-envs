#!/usr/bin/env python3
"""
Examine temporal reasoning questions in LongHealth dataset
"""

import json

def find_temporal_questions():
    with open('environments/longhealth/benchmark_v5.json', 'r') as f:
        data = json.load(f)
    
    temporal_keywords = ['first', 'last', 'before', 'after', 'chronological', 'sequence', 'order', 'timeline', 'initially', 'finally', 'subsequently', 'previously']
    
    print("=== TEMPORAL REASONING QUESTIONS IN LONGHEALTH ===")
    print()
    
    temporal_questions = []
    
    for patient_id, patient_data in data.items():
        for question in patient_data['questions']:
            question_text = question['question'].lower()
            if any(keyword in question_text for keyword in temporal_keywords):
                temporal_questions.append({
                    'patient_id': patient_id,
                    'question': question['question'],
                    'correct': question['correct']
                })
    
    print(f"Found {len(temporal_questions)} temporal reasoning questions out of 400 total questions")
    print()
    
    # Show first 10 examples
    print("SAMPLE TEMPORAL REASONING QUESTIONS:")
    for i, q in enumerate(temporal_questions[:10]):
        print(f"{i+1}. Patient {q['patient_id']}:")
        print(f"   Question: {q['question']}")
        print(f"   Correct Answer: {q['correct']}")
        print()
    
    if len(temporal_questions) > 10:
        print(f"... and {len(temporal_questions)-10} more temporal questions")
    
    print("\nTEMPORAL KEYWORDS FOUND:")
    keyword_counts = {}
    for q in temporal_questions:
        for keyword in temporal_keywords:
            if keyword in q['question'].lower():
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {keyword}: {count} questions")

if __name__ == "__main__":
    find_temporal_questions()

