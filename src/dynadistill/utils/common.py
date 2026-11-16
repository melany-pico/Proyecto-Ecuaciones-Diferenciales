from pathlib import Path
import random
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[3]
def config():
    return yaml.safe_load((ROOT/'configs/default.yml').read_text(encoding='utf-8'))
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
