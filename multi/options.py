import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gpu_id', type=int, default=3)
parser.add_argument('--height', type=int, default=28)
parser.add_argument('--width', type=int, default=28)
parser.add_argument('--model_name', type = str)
args = parser.parse_args()