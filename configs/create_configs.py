import itertools, yaml, json, os

with open("sweep.yaml") as f:
    grid = yaml.safe_load(f)

keys = grid.keys()
values = grid.values()

os.makedirs("generated", exist_ok=True)

for i, combo in enumerate(itertools.product(*values)):
    config = dict(zip(keys, combo))
    config['run_id'] = i
    with open(f"generated/run_{i}.json", "w") as f:
        json.dump(config, f, indent=2)

print(f"Generated {i+1} configs")
