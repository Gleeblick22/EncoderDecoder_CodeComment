import json
from statistics import mean

def analyze_test_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_entries = len(data)
    avg_code_len = mean([len(entry['code'].split()) for entry in data]) if data else 0
    avg_comment_len = mean([len(entry['comment'].split()) for entry in data]) if data else 0
    missing_names = sum([1 for entry in data if not entry.get('name')])

    return {
        'total_entries': total_entries,
        'avg_code_tokens': round(avg_code_len, 2),
        'avg_comment_tokens': round(avg_comment_len, 2),
        'missing_name_fields': missing_names
    }

old_test_path = "/mnt/c/ICT Research/Thesis/Project Code/T5 model/Only JAVA project source code/Datasets/test_data.json"
new_test_path = "/mnt/c/ModelTrain/Econder-Decoder/data_collection/java_only/test_data.json"


old_stats = analyze_test_file(old_test_path)
new_stats = analyze_test_file(new_test_path)

print("=== OLD TEST DATA ===")
for k, v in old_stats.items():
    print(f"{k}: {v}")

print("\n=== NEW TEST DATA ===")
for k, v in new_stats.items():
    print(f"{k}: {v}")
