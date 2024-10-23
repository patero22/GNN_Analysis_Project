import pandas as pd
import matplotlib.pyplot as plt
#test
# Funkcja do generowania wykresów
def generate_plots(results):
    # Konwertujemy wyniki na DataFrame dla łatwego tworzenia wykresów
    df = pd.DataFrame(results,
                      columns=["Model", "Library", "Format", "Dataset", "Training Time (s)", "Memory Usage (MB)",
                               "Validation Accuracy"])

    # Wykresy dla różnych parametrów
    # 1. Wykres czasu trenowania
    plt.figure(figsize=(10, 6))
    for lib in df['Library'].unique():
        subset = df[df['Library'] == lib]
        plt.plot(subset['Format'], subset['Training Time (s)'], marker='o', label=f"{lib} (CPU)")
    plt.title("Training Time for Different Formats on CPU")
    plt.xlabel("Sparse Matrix Format")
    plt.ylabel("Training Time (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig('training_time_cpu.png')
    plt.show()

    # 2. Wykres zużycia pamięci
    plt.figure(figsize=(10, 6))
    for lib in df['Library'].unique():
        subset = df[df['Library'] == lib]
        plt.plot(subset['Format'], subset['Memory Usage (MB)'], marker='o', label=f"{lib} (CPU)")
    plt.title("Memory Usage for Different Formats on CPU")
    plt.xlabel("Sparse Matrix Format")
    plt.ylabel("Memory Usage (MB)")
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_usage_cpu.png')
    plt.show()

    # Opcjonalnie: Wykres dokładności
    plt.figure(figsize=(10, 6))
    for lib in df['Library'].unique():
        subset = df[df['Library'] == lib]
        plt.plot(subset['Format'], subset['Validation Accuracy'], marker='o', label=f"{lib} (CPU)")
    plt.title("Validation Accuracy for Different Formats on CPU")
    plt.xlabel("Sparse Matrix Format")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_accuracy_cpu.png')
    plt.show()
