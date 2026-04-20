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

<img src="https://github.com/user-attachments/assets/35815ffc-db8a-40ca-98c0-98a7d18cb726" 
     alt="Structure of a neural network" 
     width="50%"/>

<p><em>Figure 2: Structure of a neural network. Image from Wikipedia.</em></p>

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
```
## Example Code: Model Prediction

Here's how to test the trained model on a single image:

```python
model.eval()
sample_image, sample_label = next(iter(train_loader))
sample_image = sample_image[0]

with torch.no_grad():
    output = model(sample_image.unsqueeze(0))
    prediction = torch.argmax(output, dim=1).item()
    probabilities = torch.softmax(output, dim=1).squeeze().numpy()

print(f"True label: {sample_label.item()}")
print(f"Predicted: {prediction}")
print(f"Confidence: {probabilities[prediction]:.2%}")
```
## Model Prediction Results Example

After training, the model correctly identifies handwritten digits:

![Model prediction output](<img width="766" height="790" alt="image" src="https://github.com/user-attachments/assets/e9947e4a-4b1f-442e-ae4d-96a7a6efc589" />)

*Example images with their predictions: the model gets 8 out of 9 correct.*

7. How Do Neural Networks Learn?

Learning = adjusting weights.

Steps:
1. Make a prediction
2. Compare with correct answer
3. Compute error
4. Update weights to reduce error

Repeat many times.

<img width="720" height="398" alt="Backpropagation in Neural Networks" src="https://github.com/user-attachments/assets/a40f9271-c48c-47be-be7c-1702cee8cfae" />

*Backpropagation in Neural Networks. Source: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2023/01/gradient-descent-vs-backpropagation-whats-the-difference/).*
