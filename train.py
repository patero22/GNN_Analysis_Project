import torch.optim as optim
import torch.nn.functional as F
import time
import memory_profiler
import torch
from torch.profiler import profile, ProfilerActivity, record_function
#test

def train(model, data, analyze_with_profiler=False, profiler_name="trace.json"):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    start_time = time.time()
    mem_usage_before = memory_profiler.memory_usage()[0]

    if analyze_with_profiler:
        # Tworzymy unikalny plik dla każdego wywołania
        trace_file = f"profiler_{profiler_name}.json"

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for epoch in range(1, 3):  # Analizujemy tylko pierwsze 2 epoki
                with record_function("model_training"):
                    optimizer.zero_grad()
                    out = model(data)

                    # PyG: używamy data.train_mask i data.y
                    if hasattr(data, 'train_mask'):
                        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                    else:
                        # DGL: Zakładamy, że dane są w jednorodnym formacie
                        train_mask = data.ndata['train_mask']
                        label = data.ndata['label']
                        if isinstance(train_mask, dict):
                            train_mask = train_mask['_N']
                            label = label['_N']
                        loss = F.nll_loss(out[train_mask], label[train_mask])

                    loss.backward()
                    optimizer.step()
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # Eksportowanie śladu do pliku
        try:
            prof.export_chrome_trace(trace_file)
            print(f"Profiling trace saved to {trace_file}")
        except RuntimeError as e:
            print(f"Warning: Could not save profiling trace to {trace_file}. Error: {e}")

    else:
        for epoch in range(20):
            optimizer.zero_grad()
            out = model(data)

            if hasattr(data, 'train_mask'):
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            else:
                train_mask = data.ndata['train_mask']
                label = data.ndata['label']
                if isinstance(train_mask, dict):
                    train_mask = train_mask['_N']
                    label = label['_N']
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

def profile_training_step(model, data):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA  # Jeśli używasz GPU
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # Zapisywanie wyników do pliku
        record_shapes=True,  # Zapisujemy również kształty tensorów
        with_stack=True  # Analiza stosu wywołań (opcjonalne)
    ) as prof:
        # Jedna iteracja treningu
        optimizer.zero_grad()
        with record_function("forward_pass"):
            out = model(data)

        # Zależne od używanej biblioteki
        if hasattr(data, 'train_mask'):  # PyG
            target = data.y
            if target.dim() > 1:
                target = target.squeeze(dim=-1)
            loss = F.nll_loss(out[data.train_mask], target[data.train_mask])
        else:  # DGL
            train_mask = data.ndata['train_mask']
            label = data.ndata['label']
            if label.dim() > 1:
                label = label.squeeze(dim=-1)
            loss = F.nll_loss(out[train_mask], label[train_mask])

        with record_function("backward_pass"):
            loss.backward()

        optimizer.step()

    # Drukujemy skrót wyników w konsoli
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Zapis profilu do pliku
    prof.export_chrome_trace("trace.json")

    return prof




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
