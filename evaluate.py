import yaml
def benchmark(config):
    path="configs/evals/green.yaml"
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    