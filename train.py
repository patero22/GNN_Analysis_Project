import torch.optim as optim
import torch.nn.functional as F
import time
import memory_profiler


def train(model, data):
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()

    start_time = time.time()
    mem_usage_before = memory_profiler.memory_usage()[0]

    for epoch in range(200):
        optimizer.zero_grad()

        out = model(data)

        if hasattr(data, 'train_mask'):
            # PyG
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        else:
            # DGL
            loss = F.nll_loss(out[data.ndata['train_mask']], data.ndata['label'][data.ndata['train_mask']])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

    end_time = time.time()
    mem_usage_after = memory_profiler.memory_usage()[0]

    train_time = end_time - start_time
    mem_usage = mem_usage_after - mem_usage_before

    return train_time, mem_usage
