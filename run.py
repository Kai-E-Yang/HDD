from dataload import load_json_config
from train import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args = parser.parse_args()

print(args.config)

config = load_json_config(custom_path=args.config)
print(config)

main(config)
