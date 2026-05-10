import torch
import pickle


data_dir = '../data/'
material = 'Fe'
with open(data_dir + f'histograms_G4_{material}.pkl', 'rb') as f:
    histograms = pickle.load(f)
