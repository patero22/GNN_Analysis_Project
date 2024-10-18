import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import memory_profiler


# Funkcja trenowania z pomiarami czasu i pamięci
def train(model, data):
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()

    # Pomiar zużycia pamięci przed rozpoczęciem
    mem_usage_before = memory_profiler.memory_usage()[0]

    start_time = time.time()  # Zaczynamy pomiar czasu
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

    end_time = time.time()  # Zakończ pomiar czasu

    # Pomiar zużycia pamięci po zakończeniu
    mem_usage_after = memory_profiler.memory_usage()[0]

    training_time = end_time - start_time
    mem_usage_diff = mem_usage_after - mem_usage_before

    print(f"Czas trenowania: {training_time:.2f} sekund")
    print(f"Zużycie pamięci: {mem_usage_diff:.2f} MB")

    return training_time, mem_usage_diff
