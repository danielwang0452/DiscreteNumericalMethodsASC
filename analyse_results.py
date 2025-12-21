import json
from pathlib import Path

json_dir = Path("results_8x4/")

results = {}
#results_string = \
#    f'{method}-{epoch}-{args.optimiser_name}-' \
#    f'{categorical_dim}x{latent_dim}-{temperature}-{lr}-{seed}'
for json_path in json_dir.glob("*.json"):
    with open(json_path) as f:
        results[json_path.stem] = json.load(f)
print(results.keys())