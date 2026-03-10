import json

input_file = "Sampled-VM14K.jsonl"      
output_file = "eastern_medicine_VM14K.jsonl"

results = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if "Eastern Medicine" in item.get("medical_topic", []):
            results.append(item)

# ghi ra file mới
with open(output_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Found {len(results)} samples with label 'Eastern Medicine'")