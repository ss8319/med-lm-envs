import json

# Pretty print metadata.json
print("=== METADATA.JSON ===")
with open('environments/longhealth/outputs/evals/longhealth--gpt-4.1-mini/d9ff7eb9/metadata.json', 'r') as f:
    metadata = json.load(f)
    print(json.dumps(metadata, indent=2))

print("\n" + "="*80 + "\n")

# Pretty print results.jsonl (each line is a separate JSON object)
print("=== RESULTS.JSONL ===")
with open('environments/longhealth/outputs/evals/longhealth--gpt-4.1-mini/d9ff7eb9/results.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        if line.strip():  # Skip empty lines
            print(f"--- Entry {i} ---")
            data = json.loads(line)
            print(json.dumps(data, indent=2))
            print()
