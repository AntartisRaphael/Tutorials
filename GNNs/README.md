# Introduction to Graph Neural Networks (GNNs)

## 1. Motivation

Many real-world data structures are **not grids** like images.

Examples:
- Social networks (users connected to friends)
- Molecules (atoms connected by bonds)
- Transportation networks (cities connected by roads)

These are naturally represented as **graphs**, not images or sequences.

---

## 2. What is a Graph?

A graph is defined as:

$$G = (V, E)$$

where:
- $V$ is the set of nodes (vertices)
- $E$ is the set of edges (connections between nodes)

<p align="center">
  <a href="https://www.puppygraph.com/blog/social-network-graphs">
    <img src="https://github.com/user-attachments/assets/2336302e-6706-46d8-a891-9cee978d7d9f" 
         width="700" 
         alt="Social graph example">
  </a>
</p>

---

## 3. Graph Neural Networks and Node Features

A **Graph Neural Network (GNN)** learns from graph-structured data.

**Core idea:** Each node updates its representation by **aggregating information from its neighbors**.

<p align="center">
  <a href="https://blogs.nvidia.com/blog/what-are-graph-neural-networks">
    <img src="https://github.com/user-attachments/assets/32252bbc-abdb-4cd3-9bf7-20af371794e5" 
         width="700" 
         alt="GNN message passing illustration">
  </a>
</p>

<p align="center">
  <em><strong>Figure 1:</strong> Message passing in a Graph Neural Network. Each node aggregates information from its neighbors to update its representation.</em>
</p>

---

To perform this aggregation, each node is associated with a feature vector.

Each node $u \in V$ has a feature vector $x_u \in \mathbb{R}^d$.

Examples:
- Social networks → age, interests  
- Molecules → atom type, charge  

All node features are collected into a matrix:

$$
X = \begin{bmatrix}
x_1^\top \\
x_2^\top \\
\vdots \\
x_n^\top
\end{bmatrix} \in \mathbb{R}^{n \times d}
$$

Each row corresponds to one node's features.
---

## 4. Common GNN Tasks

Graph Neural Networks can be applied to different types of prediction tasks depending on what we want to learn from the graph.

### Node Classification
- Predict a label for each node in the graph  
- Example: Classifying users in a social network (e.g., interests or roles)

### Link Prediction
- Predict whether an edge exists between two nodes  
- Example: Recommending new friends in a social network  

### Graph Classification
- Predict a label for an entire graph  
- Example: Classifying molecules as toxic or non-toxic  

<p align="center">
  <a href="https://link.springer.com/article/10.1186/s40537-023-00876-4">
    <img src="https://github.com/user-attachments/assets/b77c5af3-0e2b-4b79-a38f-34d53c1dcdd7" 
         width="600" 
         alt="GNN tasks overview">
  </a>
</p>

<p align="center">
  <em><strong>Figure 3:</strong> GNN tasks overview. Source: 
  <a href="https://link.springer.com/article/10.1186/s40537-023-00876-4">Springer</a>.
  </em>
</p>

---

## 5. Message Passing: The GCN Layer

A **Graph Convolutional Network (GCN)** layer updates all node embeddings as:

$$
H^{(k)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(k-1)} W^{(k)} \right)
$$

Where:
- $H^{(k-1)} \in \mathbb{R}^{n \times d_{k-1}}$ is the input embedding matrix (for $k=1$, $H^{(0)} = X$)
- $W^{(k)} \in \mathbb{R}^{d_{k-1} \times d_k}$ is a learnable weight matrix
- $\sigma$ is an activation function (e.g., ReLU)
- $\tilde{A} = A + I$ is the adjacency matrix with self-loops
- $\tilde{D}$ is the degree matrix of $\tilde{A}$

### Per-node formulation (intuitive):

Each node $v$ updates its representation by looking at its neighbors:

$$
h_v^{(k)} = \sigma \left( \sum_{u \in \mathcal{N}(v)} W \cdot h_u^{(k-1)} \right)
$$

where $\mathcal{N}(v)$ = neighbors of node $v$.

After multiple layers, nodes capture **local + global structure**.

---

## 6. Making Predictions: The Decoder

Once we have final node embeddings $Z = H^{(K)}$ (rows $z_u$, $z_v$), we can make predictions.

For **link prediction** (predicting if an edge exists between two nodes):

$$
\hat{y}_{uv} = \sigma(z_u^\top z_v)
$$

where:
- $z_u, z_v$ are node embeddings
- $\sigma$ is the sigmoid function

**Intuition:**
- If two nodes have similar embeddings → high probability of connection
- If embeddings differ → low probability

---

## 7. Complete Pipeline

| Step | Operation | Purpose |
|------|-----------|---------|
| **Input** | $X$ (node features) + $A$ (adjacency) | Initial information |
| **Propagation** | GCN layers ($H^{(k)}$) | Mix information across neighbors |
| **Output** | Node embeddings $Z$ | Learned representations |
| **Prediction** | $\sigma(z_u^\top z_v)$ | Link existence probability |

### Example code in Python using PyTorch Geometric

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# -------------------------
# 1. Create a small graph
# -------------------------
# Edge list (undirected graph)
edge_index = torch.tensor([
    [0, 1, 2, 0],
    [1, 0, 1, 2]
], dtype=torch.long)

# Node features (3 nodes, 4 features each)
x = torch.randn((3, 4))

data = Data(x=x, edge_index=edge_index)

# -------------------------
# 2. Define GNN encoder
# -------------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # node embeddings

# -------------------------
# 3. Link prediction decoder
# -------------------------
def decode(z, edge_pairs):
    # Dot product decoder
    return (z[edge_pairs[0]] * z[edge_pairs[1]]).sum(dim=1)

# -------------------------
# 4. Training setup
# -------------------------
model = GCN(in_channels=4, hidden_channels=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Positive edges (existing)
pos_edge_index = edge_index

# Negative edges (non-existing)
neg_edge_index = torch.tensor([
    [1, 2],
    [2, 0]
], dtype=torch.long)

# -------------------------
# 5. Training loop
# -------------------------
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    z = model(data.x, data.edge_index)

    # Positive predictions
    pos_pred = decode(z, pos_edge_index)
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_pred, torch.ones(pos_pred.size(0))
    )

    # Negative predictions
    neg_pred = decode(z, neg_edge_index)
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_pred, torch.zeros(neg_pred.size(0))
    )

    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------------------------
# 6. Inference
# -------------------------
model.eval()
z = model(data.x, data.edge_index)

# Predict probability of edge (0,2)
edge = torch.tensor([[0], [2]])
score = torch.sigmoid(decode(z, edge))

print(f"Link probability (0,2): {score.item():.4f}")
```
---

## 8. CNN vs GNN

| Feature | Convolutional Neural Network (CNN) | Graph Neural Network (GNN) |
|---------|------------------------------------|-----------------------------|
| Data type | Grid (images) | Graph (nodes + edges) |
| Structure | Fixed (e.g., 3×3 patches) | Irregular (varying connections) |
| Operation | Convolution (sliding filter) | Message passing (aggregate neighbors) |
| Weight sharing | Yes | Yes |
| Example use | Image recognition | Social networks, molecules |

**CNN = structured locality** (nearby pixels in a grid)

**GNN = relational locality** (connected nodes in a graph)

---

## 9. Example Applications

Graph Neural Networks are widely used in real-world applications:

- **Social networks** → friend recommendations (link prediction)  
- **Chemistry** → molecule property prediction (graph classification)  
- **Knowledge graphs** → relation prediction  
- **Fraud detection** → identifying suspicious connections  

### Netflix Recommendation System

A well-known example of recommendation systems is the **Netflix Prize**, a machine learning competition launched in 2006.  

Netflix challenged researchers to improve its recommendation algorithm (Cinematch) by predicting user ratings for movies. The competition offered a **$1 million prize** and used a dataset of over **100 million user ratings**. :contentReference[oaicite:0]{index=0}  

The goal was to better model user preferences and recommend content more accurately. This challenge significantly advanced techniques such as **collaborative filtering** and **matrix factorization**, which are still widely used today. :contentReference[oaicite:1]{index=1}  

Modern systems (including Netflix today) combine multiple approaches and leverage user behavior, content features, and deep learning to generate personalized recommendations. :contentReference[oaicite:2]{index=2}  

<p align="center">
  <a href="https://pyimagesearch.com/2023/07/03/netflix-movies-and-series-recommendation-systems/">
    <img src="https://github.com/user-attachments/assets/701fe508-e71b-46bc-a1d1-f46c9381bbde" 
         width="600" 
         alt="Netflix recommendation system illustration">
  </a>
</p>

<p align="center">
  <em><strong>Figure 4:</strong> Netflix algorithm challenge. Source: 
  <a href="https://pyimagesearch.com/2023/07/03/netflix-movies-and-series-recommendation-systems/">
    PyImageSearch
  </a>.
  </em>
</p>
---

## 10. Summary

GNNs:
- Extend neural networks to graph data
- Learn from relationships between nodes
- Use **message passing** instead of convolution
- Are powerful for non-grid, relational data

---
