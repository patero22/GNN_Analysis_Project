import torch.optim as optim
import torch.nn.functional as F
import time
import memory_profiler
import torch
#OGB

def train(model, data):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    start_time = time.time()
    mem_usage_before = memory_profiler.memory_usage()[0]

    for epoch in range(50):
        optimizer.zero_grad()

        out = model(data)

        if hasattr(data, 'train_mask'):  # PyG
            target = data.y
            if target.dim() > 1:
                target = target.squeeze(dim=-1)  # Zredukowanie wymiaru
            loss = F.nll_loss(out[data.train_mask], target[data.train_mask])

        else:  # DGL
            train_mask = data.ndata['train_mask']
            label = data.ndata['label']

            if isinstance(train_mask, dict):
                train_mask = train_mask['_N']
                label = label['_N']

            # Upewnij się, że `label` jest jednowymiarowe
            if label.dim() > 1:
                label = label.squeeze(dim=-1)

            loss = F.nll_loss(out[train_mask], label[train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    end_time = time.time()
    mem_usage_after = memory_profiler.memory_usage()[0]

    train_time = end_time - start_time
    mem_usage = mem_usage_after - mem_usage_before
    print(f"Czas trenowania: {train_time:.2f} s, Zużycie pamięci: {mem_usage:.2f} MB")
    return train_time, mem_usage



def evaluate_model(model, data):
    model.eval()  # Tryb ewaluacji
    with torch.no_grad():
        out = model(data)
        if hasattr(data, 'test_mask'):  # PyG
            pred = out.argmax(dim=1)
            correct = pred[data.test_mask] == data.y[data.test_mask]
            accuracy = int(correct.sum()) / int(data.test_mask.sum())
        else:  # DGL
            test_mask = data.ndata['test_mask']
            label = data.ndata['label']

            # Sprawdzenie, czy są to słowniki (np. dla heterogenicznego grafu)
            if isinstance(test_mask, dict):
                test_mask = test_mask['_N']
                label = label['_N']

            pred = out.argmax(dim=1)
            correct = pred[test_mask] == label[test_mask]
            accuracy = int(correct.sum()) / int(test_mask.sum())
    return accuracy
