import json
from pathlib import Path

json_dir = Path("results_8x4")

all_configs = {}

for json_path in json_dir.glob("*.json"):
    with open(json_path) as f:
        all_configs[json_path.stem] = json.load(f)

# average the 10 seeds
avg_results = {}
for key, value in all_configs.items():
    average = [0, 0]
    print(key)
    for k, v in value.items():
        print(v)
        average[0] +=  0.1 * v[0]
        average[1] += 0.1 * v[1]
    avg_results[key] = average
print(avg_results)