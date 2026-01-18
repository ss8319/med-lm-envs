import json

# Pretty print metadata.json
print("=== METADATA.JSON ===")
with open('environments/longhealth/outputs/evals/longhealth--gpt-4.1-mini/d9ff7eb9/metadata.json', 'r') as f:
    metadata = json.load(f)
    print(json.dumps(metadata, indent=2))

print("\n" + "="*80 + "\n")

# Pretty print results.jsonl (summary only)
print("=== RESULTS.JSONL SUMMARY ===")
with open('environments/longhealth/outputs/evals/longhealth--gpt-4.1-mini/d9ff7eb9/results.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        if line.strip():  # Skip empty lines
            data = json.loads(line)
            print(f"--- Entry {i} ---")
            print(f"ID: {data['id']}")
            print(f"Task: {data['task']}")
            print(f"Answer: {data['answer']}")
            print(f"Reward: {data['reward']}")
            print(f"Exact Match Reward: {data['exact_match_reward']}")
            print(f"Generation Time: {data['generation_ms']:.1f}ms")
            print(f"Scoring Time: {data['scoring_ms']:.1f}ms")
            print(f"Total Time: {data['total_ms']:.1f}ms")
            print(f"Info: {json.dumps(data['info'], indent=2)}")
            print(f"Completion: {data['completion'][0]['content']}")
            print()
