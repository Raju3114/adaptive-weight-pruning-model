# Tredence AI Engineer Case Study

## Problem Title

Self-Pruning Neural Network using PyTorch

This project implements a neural network that learns to prune its own unnecessary weights during training instead of performing pruning after training.

The objective is to reduce model complexity while maintaining strong classification performance on the CIFAR-10 dataset.

---

## Project Overview

Traditional pruning usually happens after model training.

Example:

Train Model → Prune Later

In this case study, pruning happens during training itself.

Example:

Train Model + Learn Which Weights to Remove Automatically

This is achieved using a custom layer called `PrunableLinear`, where every weight has an associated learnable gate parameter.

---

## Core Idea

Each weight has:

* Normal trainable weight
* Learnable gate score

The gate score is passed through a sigmoid:

```python
Gate = sigmoid(gate_score)
```

Final effective weight becomes:

```python
Final Weight = Weight * Gate
```

If the gate becomes close to 0, that weight is effectively pruned.

If the gate stays close to 1, the weight remains active.

---

## Why L1 Regularization Creates Sparsity

The model uses the following total loss:

```python
Total Loss = Classification Loss + λ * Sparsity Loss
```

Where:

```python
Sparsity Loss = Sum of all gate values
```

This behaves like L1 regularization.

L1 regularization encourages values to move toward zero.

Since the gate values control whether a weight stays active or gets pruned, minimizing this term pushes many gates toward zero, resulting in a sparse network.

A larger lambda (λ) causes stronger pruning but may reduce accuracy.

A smaller lambda preserves accuracy but prunes less.

---

## Files Included

### self_pruning_network.py

Contains:

* Custom `PrunableLinear` implementation
* Full neural network architecture
* CIFAR-10 training loop
* Sparsity regularization loss
* Accuracy evaluation
* Sparsity percentage calculation
* Gate value distribution plot
* Lambda comparison for multiple pruning strengths

### Generated Plot Images

* gate_distribution_lambda_0.001.png
* gate_distribution_lambda_0.01.png
* gate_distribution_lambda_0.1.png

These plots visualize the distribution of gate values after training for different lambda values.

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* Matplotlib

---

## How to Run

### Step 1: Install Dependencies

```bash
pip install torch torchvision matplotlib
```

### Step 2: Run the Project

```bash
python self_pruning_network.py
```

This will:

* Download CIFAR-10 dataset
* Train the model for multiple lambda values
* Evaluate test accuracy
* Calculate sparsity percentage
* Generate gate distribution plots
* Print final comparison table

---

## Lambda Values Tested

The project compares:

* λ = 0.001
* λ = 0.01
* λ = 0.1

This demonstrates the tradeoff between:

Accuracy vs Sparsity

---

## Expected Output Table

| Lambda |         Test Accuracy |            Sparsity % |
| ------ | --------------------: | --------------------: |
| 0.001  | (generated after run) | (generated after run) |
| 0.01   | (generated after run) | (generated after run) |
| 0.1    | (generated after run) | (generated after run) |

After running the code, replace these placeholder values with your actual output.

Example:

| Lambda | Test Accuracy | Sparsity % |
| ------ | ------------: | ---------: |
| 0.001  |          72.4 |       18.6 |
| 0.01   |          68.9 |       47.2 |
| 0.1    |          55.1 |       81.7 |

---

## Design Decisions

### Why MLP instead of CNN?

A smaller MLP was intentionally used instead of a larger CNN like ResNet to keep the focus on the pruning mechanism rather than model complexity.

The goal of this assignment is correctness of self-pruning logic, not maximum CIFAR-10 accuracy.

### Why Sigmoid Gates?

Sigmoid keeps gate values between 0 and 1, making pruning behavior stable and interpretable.

### Why Three Lambda Values?

This helps clearly demonstrate how stronger regularization increases sparsity while potentially reducing performance.

---

## Future Improvements

With more time, the following can be added:

* CNN-based self-pruning architecture
* Better threshold tuning
* Structured neuron pruning
* Dynamic pruning schedules
* TensorBoard monitoring
* Better visualization dashboards
* GPU optimization for faster experiments

---

## Submission Notes

This project was built to satisfy the Tredence AI Engineering Internship Case Study requirements:

* Custom prunable layer
* Sparsity-aware training
* CIFAR-10 evaluation
* Lambda comparison
* Result reporting
* Gate distribution visualization

The focus was on clean implementation, correctness, and practical engineering clarity.

---

## Final Deliverables Submitted

This repository includes:

* Python source code
* Markdown report (README)
* Generated gate distribution plots

This completes the required case study submission for the Tredence AI Engineering Internship.
