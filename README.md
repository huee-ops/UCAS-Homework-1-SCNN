# UCAS-Homework-1-SCNN

2025年秋季国科大《GPU架构与编程》大作业一：SNN 推理优化

## 1. 项目简介
本项目是《GPU架构与编程》课程的大作业一，目标是基于 CUDA 实现并优化 Spiking Convolutional Neural Network (SCNN) 在 FashionMNIST 数据集上的推理过程。

通过引入 **Mega-Kernel（超大内核）**、**Shared Memory 全驻留** 以及 **SNN 稀疏性计算** 等优化策略，本项目实现了极致的推理性能。

**最终性能 (Final Performance)：**
* **Accuracy**: 89.78%
* **Time**: 0.0585s (Total inference time for 10k images)

## 2. 核心优化 (Optimizations)

### 2.1 算子融合与 Mega-Kernel (Kernel Fusion)
针对 SNN 多时间步（T=8）和小规模网络的特点，将 Conv1-Pool1-Conv2-Pool2-FC1-FC2-FC3 的所有计算层及时间循环融合进**单个 Kernel** (`scnn_inference_mega_kernel`)。
* **One Block per Image**: 每个 Thread Block 独立处理一张图片。
* **Zero Intermediate Global Access**: 除了输入图片和最终结果，中间层数据完全不经过 Global Memory。

### 2.2 全片上内存驻留 (Shared Memory Residence)
利用 Shared Memory 高带宽特性，将网络所有中间层的 Feature Maps (Spikes) 和神经元状态 (Membrane Potential) 全部存储在 Shared Memory 中。
* **Shared Memory Usage**: 约 45KB / Block (适配大多数现代 GPU)。

### 2.3 稀疏性感知计算 (Sparsity Optimization)
利用 SNN 脉冲的二值特性（0 或 1），在卷积和全连接层实现稀疏计算逻辑。
* **Skip Zeros**: 仅当输入脉冲非零时才加载权重进行累加，大幅减少无效运算和访存。

## 3. 编译与运行 (Build & Run)

### 环境依赖
* CUDA Toolkit 11.0+
* C++11 Compiler

### 编译
```bash
# 请根据你的 GPU 架构修改 -arch 参数 (例如: sm_75 for T4, sm_80 for A100)
nvcc -O3 -arch=sm_75 -o scnn_inference inference.cu
