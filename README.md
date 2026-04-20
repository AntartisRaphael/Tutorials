# Introduction to Neural Networks

## 1. Motivation: Why Neural Networks?

Think about problems like:

- Recognizing objects in images (cat vs dog)
- Converting speech to text
- Recommending movies or products

These are hard to solve with traditional programming because:
- The rules are too complex
- The patterns are hidden in data

**Key idea:**  
Instead of writing rules → we let the computer *learn from data*

---

## 2. What is a Neural Network?

A neural network is a **function approximator**:
- Input → Neural Network → Output


Example:
- Input: image pixels  
- Output: "cat" or "dog"

---

## 3. Biological Inspiration

Neural networks are inspired by the brain.

| Biological Neuron | Artificial Neuron |
|------------------|------------------|
| Dendrites receive signals | Inputs |
| Cell body processes | Weighted sum |
| Axon sends output | Activation function |

---

## 4. The Artificial Neuron

A single neuron computes:
y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)

![Representation of an artificial neuron and its elements](https://github.com/user-attachments/assets/1efcad8b-2b36-40de-a6f1-21fff34baf43)

*Figure 1: Representation of an artificial neuron and its elements. Geshenson 2003. Illustration by author.*

## 5. From One Neuron to a Network

A neural network is made of layers:

- **Input layer** → receives data  
- **Hidden layers** → extract patterns  
- **Output layer** → produces result

Input → Hidden Layers → Output
Each layer learns more abstract features

Simple Example Using PyTorch

This example shows a small neural network that learns to recognize handwritten digits from the MNIST dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Load dataset
transform = transforms.ToTensor()

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 2. Define neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop (1 epoch for demo)
for images, labels in train_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training step complete!")
