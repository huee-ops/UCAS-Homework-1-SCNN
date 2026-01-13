# UCAS-Homework-1-SCNN

**2025年秋季国科大《GPU架构与编程》大作业一：基于 CUDA 的 SNN 推理优化**

> **注意**：本 README 同时也作为本次大作业的说明文档。

---

## 1. 项目基本信息 (Project Info)

* **课程名称**：GPU架构与编程
* **开课学期**：2025年秋季
* **学生姓名**：
* **学生学号**：
* **提交日期**：2026年1月12日
* **开源链接**：[https://github.com/huee-ops/UCAS-Homework-1-SCNN](https://github.com/huee-ops/UCAS-Homework-1-SCNN)

---

## 2. 性能指标 (Performance)

在 FashionMNIST 测试集（10,000 张图片）上的最终测试结果：

* **测试集准确率 (Accuracy)**：**89.78%**
* **总推理耗时 (Total Inference Time)**：**0.0585s**

---

## 3. 核心优化策略 (Optimization Strategy)

本项目针对 SNN（脉冲神经网络）的时间步循环特性和 GPU 硬件架构，实施了以下三大核心优化，实现了极致的推理速度：

### 3.1 算子融合与 Mega-Kernel 架构 (Kernel Fusion)
传统的深度学习推理往往针对每一层启动一个独立的 Kernel，导致严重的显存带宽瓶颈和 Kernel 启动延迟。
* **优化方案**：设计了 `scnn_inference_mega_kernel`，将网络的所有层（Conv1, Pool1, Conv2, Pool2, FC1, FC2, FC3）以及时间步循环（T=8）全部融合在一个 Kernel 中。
* **实现细节**：每个 Thread Block 负责一张图片的完整推理（One Block per Image）。除了输入图像和最终结果，中间数据不再写回 Global Memory。

### 3.2 全片上内存驻留 (Shared Memory Residence)
配合 Mega-Kernel 架构，本项目利用 Shared Memory 的高带宽低延迟特性，实现了中间数据的全片上存储。
* **优化方案**：精心规划 Shared Memory 布局，将所有中间层的脉冲输出（Spikes）和神经元膜电位（Vm）全部驻留在 Shared Memory 中。
* **内存布局**：`s_img` (Input) -> `s_c1` -> `s_p1` -> ... -> `s_output_sum`。
* **效果**：在推理过程中，实现了对 Global Memory 的“零”中间访问。

### 3.3 SNN 稀疏性感知计算 (Sparsity Optimization)
利用 SNN 传递信息为二值脉冲（0 或 1）的稀疏特性，大幅减少无效计算。
* **优化方案**：实现了稀疏卷积和稀疏全连接逻辑 (`device_conv_bias_if_SPARSE`)。
* **实现细节**：在计算过程中，线程会先检查输入脉冲是否为非零值。仅当 `input != 0` 时，才读取对应的权重进行累加。这有效地跳过了大量无效的乘加运算，进一步提升了吞吐量。

---

## 4. 文件结构 (File Structure)

* `inference.cu`: 核心源代码，包含 Mega-Kernel V3 实现、Host 端调度及数据加载逻辑。
* `README.md`: 项目说明文档。

---

## 5. 编译与运行 (Build & Run)

### 环境依赖
* **CUDA Toolkit**: 11.0 及以上版本
* **Compiler**: `nvcc` (支持 C++11)

### 编译命令
请根据您的 GPU 架构调整 `-arch` 参数（例如 T4 使用 `sm_75`, A100 使用 `sm_80`）。

```bash
nvcc -O3 -arch=sm_75 -o scnn_inference inference.cu
