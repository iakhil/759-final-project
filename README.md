# Auto Fine-Tuning GPU Parameters for AI Workloads

## Overview
This project focuses on accelerating the training and inference of Multi-Layer Perceptrons (MLPs) using CUDA-enabled GPUs. The aim is to leverage parallel computing capabilities of GPUs to significantly reduce the computational time required for fine-tuning MLP models, particularly for deep learning applications.

## Key Objectives
- Implement MLP forward and backward passes using CUDA.
- Compare performance against CPU-based training.
- Analyze speedup and memory utilization.
- Explore potential for real-time inference on edge devices.

## Tools & Technologies
- CUDA
- C++ / Python
- PyTorch (for baseline comparison)
- NVIDIA GPU

## Motivation
As MLPs remain foundational in many neural network architectures, optimizing their performance using GPU parallelism can lead to more efficient training pipelines and open doors to deployment in real-time systems.

## Future Scope
This project can be extended to other architectures (e.g., CNNs, RNNs), or integrated into broader GPU-accelerated ML frameworks.
