import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

class NeuralNetwork1(nn.Module):
    def __init__(self):
        super(NeuralNetwork1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork2(nn.Module):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model1 = NeuralNetwork1()
model2 = NeuralNetwork2()

learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()

optimizer1 = torch.optim.SGD(model1.parameters(), lr = learning_rate)
optimizer2 = torch.optim.SGD(model2.parameters(), lr = learning_rate)

def train_loop(dataloader, model1, model2, loss_fn, optimizer1, optimizer2):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred1 = model1(X)
        pred2 = model2(pred1)
        loss2 = loss_fn(pred2, y)

        optimizer2.zero_grad()
        pred1.retain_grad()
        loss2.backward(retain_graph = True)
        pred1_grad = pred1.grad
        optimizer2.step()

        optimizer1.zero_grad()
        pred1.backward(pred1_grad)
        optimizer1.step()

        if batch % 100 == 0:
            loss2, current = loss2.item(), batch * len(X)
            print(f"loss : {loss2:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model1, model2, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X , y in dataloader:
            pred1 = model1(X)
            pred2 = model2(pred1)
            test_loss += loss_fn(pred2, y).item()
            correct += (pred2.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error : \n Accuract : {(100*correct):>0.1f}%, Avg loss : {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------")
    train_loop(train_dataloader, model1, model2, loss_fn, optimizer1, optimizer2)
    test_loop(test_dataloader, model1, model2, loss_fn)
print("Done!")
