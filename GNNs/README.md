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

---

## 3. Node Feature Matrix

Each node $u \in V$ can have a feature vector $x_u \in \mathbb{R}^d$ (e.g., age, interests for a person; atom type, charge for a molecule).

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

## 4. What is a Graph Neural Network?

A **Graph Neural Network (GNN)** learns from graph-structured data.

**Core idea:** Each node updates its representation by **aggregating information from its neighbors**.

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

- Social networks → friend recommendations (link prediction)
- Chemistry → molecule property prediction (graph classification)
- Knowledge graphs → link prediction
- Fraud detection → suspicious connections

---

## 10. Summary

GNNs:
- Extend neural networks to graph data
- Learn from relationships between nodes
- Use **message passing** instead of convolution
- Are powerful for non-grid, relational data

---
