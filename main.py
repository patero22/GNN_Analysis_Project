from data_loader import load_data
from train import train
from models import GCN
from results import run_experiments
from plot_generator import generate_plots
from visualization import generate_plots
#OGB

results = run_experiments()
#results = run_experiments(dataset_name="ogbn-arxiv")

# Generowanie wykresu po zakończeniu eksperymentów
#generate_plots()
