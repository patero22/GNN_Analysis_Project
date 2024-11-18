import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob


def load_trace_file(trace_file):
    """Wczytuje plik trace.json i zwraca dane jako listę zdarzeń."""
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)
    return trace_data['traceEvents']


def parse_events(events):
    """Przetwarza zdarzenia na DataFrame."""
    data = []
    for event in events:
        if 'dur' in event and 'name' in event:
            data.append({
                'name': event['name'],
                'category': event.get('cat', 'unknown'),
                'start_time': event['ts'],
                'duration': event['dur'],
                'process_id': event['pid'],
                'thread_id': event['tid'],
            })
    return pd.DataFrame(data)


def plot_top_operations(df, top_n=10):
    """Generuje wykres najdłuższych operacji."""
    top_operations = df.nlargest(top_n, 'duration')
    plt.barh(top_operations['name'], top_operations['duration'])
    plt.xlabel('Duration (us)')
    plt.ylabel('Operation')
    plt.title(f'Top {top_n} Longest Operations')
    plt.show()


def plot_total_time_by_category(df):
    """Generuje wykres sumy czasu trwania operacji dla każdej kategorii."""
    total_time = df.groupby('category')['duration'].sum().sort_values(ascending=False)
    total_time.plot(kind='bar')
    plt.xlabel('Category')
    plt.ylabel('Total Time (us)')
    plt.title('Total Time by Category')
    plt.show()


def plot_duration_histogram(df):
    """Generuje histogram czasów trwania operacji."""
    plt.hist(df['duration'], bins=50, log=True)
    plt.xlabel('Duration (us)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Operation Durations')
    plt.show()


def analyze_trace_file(trace_file):
    """Główna funkcja analizy pojedynczego pliku trace.json."""
    events = load_trace_file(trace_file)
    df = parse_events(events)

    print(f"Analizuję plik: {trace_file}")
    print("Najdłuższe operacje:")
    print(df.nlargest(10, 'duration'))

    # Wykresy
    plot_top_operations(df)
    plot_total_time_by_category(df)
    plot_duration_histogram(df)


def analyze_all_traces(directory):
    """Przetwarza wszystkie pliki .trace.json w katalogu."""
    trace_files = glob(os.path.join(directory, "*.trace.json"))
    if not trace_files:
        print(f"Brak plików do analizy w katalogu: {directory}")
        return

    print(f"Znaleziono {len(trace_files)} plików do analizy.")
    for trace_file in trace_files:
        try:
            analyze_trace_file(trace_file)
        except Exception as e:
            print(f"Błąd podczas analizy pliku {trace_file}: {e}")


# Użycie
if __name__ == "__main__":
    logs_directory = "logs"  # Katalog, w którym znajdują się pliki JSON
    analyze_all_traces(logs_directory)
