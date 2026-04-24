# 🧠 Self-Pruning Neural Network

**AI Engineer Case Study**

---

## 📌 1. Introduction

Deep neural networks often contain a large number of parameters, making them computationally expensive and memory-intensive. To address this, **model pruning** is used to remove less important connections.

In this project, we implemented a **self-pruning neural network** that learns to remove unnecessary weights **during training**, instead of performing pruning after training.

---

## 🧩 2. Core Idea

Each weight in the network is associated with a **learnable gate parameter**:

* Gate values range between **0 and 1**
* If gate → 0 ⇒ weight is effectively removed
* If gate → 1 ⇒ weight remains active

### Mathematical Representation

```id="cy66pw"
gates = sigmoid(gate_scores)
pruned_weight = weight × gates
```

---

## ⚙️ 3. Methodology

### 🔹 3.1 Prunable Linear Layer

A custom layer `PrunableLinear` was implemented with:

* Weight matrix
* Bias
* Learnable gate scores

During forward pass:

* Gate scores are passed through sigmoid
* Weights are multiplied with gates
* Resulting pruned weights are used in computation

---

### 🔹 3.2 Loss Function

```id="0v6w16"
Total Loss = Classification Loss + λ × Sparsity Loss
```

Where:

* Classification Loss → Cross-Entropy
* Sparsity Loss → Mean of gate values

```id="5ia2ua"
Sparsity Loss = mean(gates)
```

---

### 🔹 3.3 Why L1 Regularization Encourages Sparsity

👉 Simple explanation:
The sparsity term penalizes active gates, forcing many of them to shrink toward zero.

---

## 🏗️ 4. Model Architecture

```id="hj5f9j"
Input (32×32×3)
      ↓
Flatten
      ↓
PrunableLinear (512)
      ↓
ReLU
      ↓
PrunableLinear (256)
      ↓
ReLU
      ↓
PrunableLinear (10)
      ↓
Output
```

---

## 🧪 5. Experimental Setup

* Dataset: CIFAR-10
* Optimizer: Adam
* Learning Rate: 0.001
* Epochs: 15
* Batch Size: 128

### Lambda Values Tested:

* 0.5 (Balanced pruning)
* 1.0 (Moderate pruning)
* 5.0 (High pruning)
* 20.0 (Extreme pruning)

---

## 📊 6. Results

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
| ---------- | ----------------- | ------------------ |
| 0.5        | 48.97             | 92.76              |
| 1.0        | 48.27             | 95.53              |
| 5.0        | 43.67             | 98.71              |
| 20.0       | 36.46             | 99.69              |

---

## 📈 7. Analysis

### 🔹 Sparsity vs Lambda

* Sparsity increases significantly with λ
* Even at λ = 0.5, over **90% weights are pruned**

---

### 🔹 Accuracy vs Lambda

* Accuracy decreases gradually as λ increases
* High λ removes important connections

---

### 🔹 Trade-off (Key Insight)

👉 The model shows a clear trade-off between:

* **Efficiency (Sparsity ↑)**
* **Performance (Accuracy ↓)**

👉 λ = 0.5 provides a **better balance** compared to extreme values.

---

## 📉 8. Gate Value Distribution

The histogram shows:

* Large spike near **0 → pruned weights**
* Small cluster near **1 → important weights**

👉 Confirms successful self-pruning behavior

---

## 🔍 9. Where Pruning Happens

```id="4uawb9"
pruned_weights = weight × gates
```

👉 If gate ≈ 0 ⇒ weight ≈ 0 ⇒ connection removed

---

## 📌 10. Observations

* The network successfully prunes itself during training
* Very high sparsity (90–99%) achieved
* Excessive pruning reduces accuracy
* Moderate λ (0.5–1.0) provides better performance

---

## ✅ 11. Conclusion

* The proposed model effectively learns to prune its own connections
* L1-based sparsity regularization drives pruning
* There is a clear trade-off between accuracy and sparsity
* Proper selection of λ is crucial for optimal results

---

## 🚀 12. Future Work

* Use CNN architectures for better accuracy
* Apply structured pruning
* Combine pruning with quantization
* Deploy on edge devices

---

## 📌 13. Final Summary

👉 A self-pruning neural network dynamically removes unnecessary connections during training using learnable gates, achieving up to **99% sparsity** while maintaining reasonable accuracy at moderate pruning levels.

---
