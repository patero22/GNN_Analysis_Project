import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#OGB
# Funkcja do generowania wszystkich wykresów na podstawie zebranych danych
def generate_plots(results_file='experiment_results.csv'):
    # Wczytanie danych z pliku CSV
    df = pd.read_csv(results_file)

    # 1. Czas trenowania w zależności od formatu macierzy
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Format', y='Training Time (s)', hue='Library', data=df, palette='Set2')
    sns.stripplot(x='Format', y='Training Time (s)', hue='Library', data=df, dodge=True, linewidth=1, palette='Set2', alpha=0.6)
    plt.title('Training Time vs Matrix Format')
    plt.ylabel('Training Time (s)')
    plt.xlabel('Matrix Format')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('training_time_vs_format.png')
    plt.show()

    # 2. Zużycie pamięci w zależności od formatu macierzy
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Format', y='Memory Usage (MB)', hue='Library', data=df, palette='Set1')
    sns.stripplot(x='Format', y='Memory Usage (MB)', hue='Library', data=df, dodge=True, linewidth=1, palette='Set1', alpha=0.6)
    plt.title('Memory Usage vs Matrix Format')
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Matrix Format')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('memory_usage_vs_format.png')
    plt.show()

    # 3. Czas trenowania dla różnych modeli
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='Training Time (s)', hue='Library', data=df, palette='muted')
    sns.stripplot(x='Model', y='Training Time (s)', hue='Library', data=df, dodge=True, linewidth=1, palette='muted', alpha=0.6)
    plt.title('Training Time vs Model')
    plt.ylabel('Training Time (s)')
    plt.xlabel('Model')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('training_time_vs_model.png')
    plt.show()

    # 4. Zużycie pamięci dla różnych modeli
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='Memory Usage (MB)', hue='Library', data=df, palette='coolwarm')
    sns.stripplot(x='Model', y='Memory Usage (MB)', hue='Library', data=df, dodge=True, linewidth=1, palette='coolwarm', alpha=0.6)
    plt.title('Memory Usage vs Model')
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Model')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('memory_usage_vs_model.png')
    plt.show()

    # 5. Dokładność w zależności od modelu
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='Validation Accuracy', hue='Library', data=df, palette='deep')
    sns.stripplot(x='Model', y='Validation Accuracy', hue='Library', data=df, dodge=True, linewidth=1, palette='deep', alpha=0.6)
    plt.title('Validation Accuracy vs Model')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Model')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('validation_accuracy_vs_model.png')
    plt.show()

    # 6. Dokładność w zależności od formatu macierzy
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Format', y='Validation Accuracy', hue='Library', data=df, palette='dark')
    sns.stripplot(x='Format', y='Validation Accuracy', hue='Library', data=df, dodge=True, linewidth=1, palette='dark', alpha=0.6)
    plt.title('Validation Accuracy vs Matrix Format')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Matrix Format')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('validation_accuracy_vs_format.png')
    plt.show()

    # 7. Zużycie pamięci w zależności od zbioru danych
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Dataset', y='Memory Usage (MB)', hue='Library', data=df, palette='cubehelix')
    sns.stripplot(x='Dataset', y='Memory Usage (MB)', hue='Library', data=df, dodge=True, linewidth=1, palette='cubehelix', alpha=0.6)
    plt.title('Memory Usage vs Dataset')
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Dataset')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('memory_usage_vs_dataset.png')
    plt.show()

    # 8. Czas trenowania w zależności od zbioru danych
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Dataset', y='Training Time (s)', hue='Library', data=df, palette='cool')
    sns.stripplot(x='Dataset', y='Training Time (s)', hue='Library', data=df, dodge=True, linewidth=1, palette='cool', alpha=0.6)
    plt.title('Training Time vs Dataset')
    plt.ylabel('Training Time (s)')
    plt.xlabel('Dataset')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('training_time_vs_dataset.png')
    plt.show()

