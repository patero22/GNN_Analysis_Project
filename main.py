from data_loader import load_data
from train import train
from models import GCN
from results import run_experiments
from plot_generator import generate_plots
from visualization import generate_plots

results = run_experiments()


# Generowanie wykresu po zakończeniu eksperymentów
generate_plots()
