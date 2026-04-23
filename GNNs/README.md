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

$$
G = (V, E)
$$

where:
- $V$ is the set of nodes (vertices)
- $E$ is the set of edges (connections between nodes)

Each node can also have **features**:
- Example: a person → age, interests  
- Example: an atom → type, charge  

---

## 3. What is a Graph Neural Network?

A Graph Neural Network (GNN) is a model that learns from graph-structured data.

Key idea:

> Each node updates its representation by aggregating information from its neighbors.

---

## 4. Message Passing (Core Mechanism)

At each layer, a node updates its features using its neighbors:

$$
h_v^{(k)} = \sigma \left( \sum_{u \in \mathcal{N}(v)} W \cdot h_u^{(k-1)} \right)
$$

where:
- $h_v^{(k)}$ = representation of node $v$ at layer $k$
- $\mathcal{N}(v)$ = neighbors of node $v$
- $W$ = learnable weights
- $\sigma$ = activation function (e.g., ReLU)

This is called **message passing**.

---

## 5. Intuition

- Each node “looks” at its neighbors
- Combines their information
- Updates its own representation

After multiple layers:
- Nodes capture **local + global structure**

---

## 6. Example Applications

- Social networks → friend recommendations
- Chemistry → molecule property prediction
- Knowledge graphs → link prediction
- Fraud detection → suspicious connections

---

# 🔍 CNN vs GNN

## Key Differences

| Feature | Convolutional Neural Network (CNN) | Graph Neural Network (GNN) |
|--------|------------------------------------|-----------------------------|
| Data type | Grid (images) | Graph (nodes + edges) |
| Structure | Fixed (e.g., pixels in a grid) | Irregular (varying connections) |
| Neighborhood | Local patches (e.g., 3×3) | Arbitrary neighbors |
| Operation | Convolution (sliding filters) | Message passing |
| Weight sharing | Yes | Yes |
| Example use | Image recognition | Social networks, molecules |

---

## Intuition Comparison

- **CNN:** looks at nearby pixels in a fixed grid  
- **GNN:** looks at connected nodes in a graph  

CNN = structured locality  
GNN = relational locality  

---

## 7. Summary

GNNs:
- Extend neural networks to graph data
- Learn from relationships between nodes
- Use **message passing** instead of convolution
- Are powerful for non-grid, relational data

---
