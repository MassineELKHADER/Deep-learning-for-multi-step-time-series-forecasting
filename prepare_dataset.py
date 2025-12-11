import numpy as np
from data.synthetic_dataset import create_synthetic_dataset
import pickle

N = 500
N_input = 20
N_output = 20
sigma = 0.01

dataset = create_synthetic_dataset(N, N_input, N_output, sigma)

with open("synthetic_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("Dataset saved!")
