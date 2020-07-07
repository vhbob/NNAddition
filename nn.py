import random
import torch.nn as nn
import torch


class NN(nn.Module):
    def __init__(self, n_ins, n_outs):
        super(NN, self).__init__()
        self.hidden = nn.Linear(n_ins, n_ins)
        self.out = nn.Linear(n_ins, n_outs)

    def forward(self, input_vals):
        return self.out(self.hidden(input_vals))


model = NN(2, 1)
criterion = nn.MSELoss()
optim = torch.optim.Adam(lr=0.000025, params=model.parameters())
for epoch in range(1, 101):
    optim.zero_grad()
    model.zero_grad()
    criterion.zero_grad()
    c_loss = 0
    for data in range(0, 1000):
        n1 = random.randint(-1000, 1000)
        n2 = random.randint(-1000, 1000)
        target = n1 + n2
        x = torch.as_tensor([n1, n2], dtype=torch.float)
        y = model(x)
        loss = criterion(y, torch.as_tensor(target, dtype=torch.float).reshape([1]))
        loss.backward()
        optim.step()
        c_loss += loss.item()
    print("loss for", epoch, c_loss/1000)

print("Whats 9+10?", model(torch.as_tensor([9, 10], dtype=torch.float))[0])
print("Whats 12312321+51234123?", model(torch.as_tensor([12312321, 51234123], dtype=torch.float))[0])
