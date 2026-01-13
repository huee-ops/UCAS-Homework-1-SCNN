#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cfloat> // For FLT_MAX

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}
// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================


// ===================================================================================
// SCNN CUDA Kernels (Mega-Kernel V3 - SNN Sparsity Optimization)
// ===================================================================================

// --- PTX IF Neuron (与之前相同) ---
__device__ __forceinline__ void if_neuron_ptx_kernel(float layer_output, float* vm_addr, float* spike_addr) {
    const float threshold = 1.0f;
    const float v_reset = 0.0f;
    const float current_vm = *vm_addr;
    const float new_vm = current_vm + layer_output;
    float spike_out, vm_out;

    asm("{\n\t"
        ".reg .pred %%p;\n\t"
        "setp.ge.f32 %%p, %2, %3;\n\t"
        "selp.f32 %0, 1.0, 0.0, %%p;\n\t"
        "selp.f32 %1, %4, %2, %%p;\n\t"
        "}"
        : "=f"(spike_out), "=f"(vm_out)
        : "f"(new_vm), "f"(threshold), "f"(v_reset)
    );
    *spike_addr = spike_out;
    *vm_addr = vm_out;
}

// --- Mega-Kernel 帮助函数 (Device) ---

/**
 * @brief (Device) 融合 (Conv2D + Bias + IFNeuron) - [DENSE Version]
 * @note 用于 Conv1，输入是连续值图像
 * 权重 (weights) 从全局内存读取
 */
__device__ void device_conv_bias_if_DENSE(
    const float* input, const float* __restrict__ weights, const float* bias,
    float* vm, float* output_spikes,
    int C_in, int H_in, int W_in,
    int C_out, int K)
{
    const int H_out = H_in - K + 1;
    const int W_out = W_in - K + 1;
    const int N_out = C_out * H_out * W_out;
    const int tid = threadIdx.x;
    
    for (int neuron_idx = tid; neuron_idx < N_out; neuron_idx += blockDim.x) {
        const int w_out = neuron_idx % W_out;
        const int h_out = (neuron_idx / W_out) % H_out;
        const int c_out = neuron_idx / (W_out * H_out);

        float conv_sum = 0.0f;

        // 2. 执行 DENSE 卷积 (带乘法)
        for (int c = 0; c < C_in; ++c) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    const int h_in_idx = h_out + kh;
                    const int w_in_idx = w_out + kw;
                    
                    const int input_idx = c * H_in * W_in + h_in_idx * W_in + w_in_idx;
                    const int weight_idx = c_out * C_in * K * K + c * K * K + kh * K + kw;

                    conv_sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
        const float layer_output = conv_sum + bias[c_out];

        if_neuron_ptx_kernel(layer_output, &vm[neuron_idx], &output_spikes[neuron_idx]);
    }
}

/**
 * @brief (Device) 融合 (Conv2D + Bias + IFNeuron) - [SPARSE Version]
 * @note 用于 Conv2，输入是二值脉冲
 * 权重 (weights) 从全局内存读取
 */
__device__ void device_conv_bias_if_SPARSE(
    const float* input, const float* __restrict__ weights, const float* bias,
    float* vm, float* output_spikes,
    int C_in, int H_in, int W_in,
    int C_out, int K)
{
    const int H_out = H_in - K + 1;
    const int W_out = W_in - K + 1;
    const int N_out = C_out * H_out * W_out;
    const int tid = threadIdx.x;
    
    for (int neuron_idx = tid; neuron_idx < N_out; neuron_idx += blockDim.x) {
        const int w_out = neuron_idx % W_out;
        const int h_out = (neuron_idx / W_out) % H_out;
        const int c_out = neuron_idx / (W_out * H_out);

        float conv_sum = 0.0f;

        // 2. [优化] 执行 SPARSE 卷积 (无乘法)
        for (int c = 0; c < C_in; ++c) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    const int h_in_idx = h_out + kh;
                    const int w_in_idx = w_out + kw;
                    
                    const int input_idx = c * H_in * W_in + h_in_idx * W_in + w_in_idx;
                    
                    // [SNN 优化] 只有当输入脉冲 != 0.0 时才累加权重
                    if (input[input_idx] != 0.0f) {
                        const int weight_idx = c_out * C_in * K * K + c * K * K + kh * K + kw;
                        conv_sum += weights[weight_idx];
                    }
                }
            }
        }
        const float layer_output = conv_sum + bias[c_out];

        if_neuron_ptx_kernel(layer_output, &vm[neuron_idx], &output_spikes[neuron_idx]);
    }
}


/**
 * @brief (Device) Max Pooling 2D (与之前相同)
 */
__device__ void device_max_pool2d(
    const float* input, float* output,
    int C, int H_in, int W_in, int K)
{
    const int H_out = H_in / K;
    const int W_out = W_in / K;
    const int N_out = C * H_out * W_out;
    const int tid = threadIdx.x;

    for (int neuron_idx = tid; neuron_idx < N_out; neuron_idx += blockDim.x) {
        const int w_out = neuron_idx % W_out;
        const int h_out = (neuron_idx / W_out) % H_out;
        const int c = neuron_idx / (W_out * H_out);

        const int h_start = h_out * K;
        const int w_start = w_out * K;
        float max_val = -FLT_MAX;

        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int h_in_idx = h_start + kh;
                const int w_in_idx = w_start + kw;
                const int input_idx = c * H_in * W_in + h_in_idx * W_in + w_in_idx;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
        output[neuron_idx] = max_val;
    }
}

/**
 * @brief (Device) 融合 (Linear + Bias + IFNeuron) - [SPARSE Version]
 * @note 用于 FC1, FC2。输入是二值脉冲
 */
__device__ void device_linear_bias_if_SPARSE(
    const float* input, const float* __restrict__ weights, const float* bias,
    float* vm, float* output_spikes,
    int N_in, int N_out)
{
    const int tid = threadIdx.x;

    for (int neuron_idx = tid; neuron_idx < N_out; neuron_idx += blockDim.x) {
        float linear_sum = 0.0f;
        
        // [优化] 执行 SPARSE 线性层
        for (int j = 0; j < N_in; ++j) {
            // [SNN 优化] 只有当输入脉冲 != 0.0 时才累加权重
            if (input[j] != 0.0f) {
                linear_sum += weights[neuron_idx * N_in + j];
            }
        }
        const float layer_output = linear_sum + bias[neuron_idx];

        if_neuron_ptx_kernel(layer_output, &vm[neuron_idx], &output_spikes[neuron_idx]);
    }
}

/**
 * @brief (Device) 融合 (Linear + Bias + Accumulate) - [SPARSE Version]
 * @note 用于 FC3。输入是二值脉冲
 */
__device__ void device_linear_bias_accum_SPARSE(
    const float* input, const float* __restrict__ weights, const float* bias,
    float* output_sum,
    int N_in, int N_out)
{
    const int tid = threadIdx.x;

    for (int neuron_idx = tid; neuron_idx < N_out; neuron_idx += blockDim.x) {
        float linear_sum = 0.0f;
        
        // [优化] 执行 SPARSE 线性层
        for (int j = 0; j < N_in; ++j) {
            // [SNN 优化] 只有当输入脉冲 != 0.0 时才累加权重
            if (input[j] != 0.0f) {
                linear_sum += weights[neuron_idx * N_in + j];
            }
        }
        
        output_sum[neuron_idx] += (linear_sum + bias[neuron_idx]);
    }
}


// --- [终极优化 V3] "超级内核" (SNN 稀疏性) ---
__global__ void scnn_inference_mega_kernel(
    const float* d_image, // 全局输入 (B, 784)
    float* d_output_sum,  // 全局输出 (B, 10)
    // 全局权重/偏置
    const float* d_conv1_w, const float* d_conv1_b, const float* d_conv2_w, const float* d_conv2_b,
    const float* d_fc1_w,   const float* d_fc1_b,   const float* d_fc2_w,   const float* d_fc2_b,
    const float* d_fc3_w,   const float* d_fc3_b,
    const int T, const int BATCH_SIZE) 
{
    // --- 0. 定义维度 (与 V1 相同) ---
    const int IMG_C = 1; const int IMG_H = 28; const int IMG_W = 28;
    const int IMG_SIZE = 784;
    const int C1_K = 5; const int C1_OUT_C = 6;
    const int C1_OUT_H = 24; const int C1_OUT_W = 24;
    const int C1_OUT_SIZE = 3456;
    const int P1_K = 2; const int P1_OUT_C = C1_OUT_C;
    const int P1_OUT_H = 12; const int P1_OUT_W = 12;
    const int P1_OUT_SIZE = 864;
    const int C2_K = 5; const int C2_OUT_C = 16;
    const int C2_OUT_H = 8; const int C2_OUT_W = 8;
    const int C2_OUT_SIZE = 1024;
    const int P2_K = 2; const int P2_OUT_C = C2_OUT_C;
    const int P2_OUT_H = 4; const int P2_OUT_W = 4;
    const int P2_OUT_SIZE = 256;
    const int FC1_OUT_SIZE = 120;
    const int FC2_OUT_SIZE = 84;
    const int FC3_OUT_SIZE = 10;
    const int VMS_OUT_SIZE = C1_OUT_SIZE + C2_OUT_SIZE + FC1_OUT_SIZE + FC2_OUT_SIZE; // 4684

    // --- 1. 索引和共享内存设置 (与 V1 相同) ---
    extern __shared__ float s_data[];

    const int batch_idx = blockIdx.x;
    if (batch_idx >= BATCH_SIZE) return;
    
    const int tid = threadIdx.x;
    
    // 共享内存划分 (V1 布局, 45128 字节)
    float* s_img            = s_data;                                         // 784
    float* s_c1_out_spikes  = s_img + IMG_SIZE;                               // 3456
    float* s_p1_out_spikes  = s_c1_out_spikes + C1_OUT_SIZE;                  // 864
    float* s_c2_out_spikes  = s_p1_out_spikes + P1_OUT_SIZE;                  // 1024
    float* s_p2_out_spikes  = s_c2_out_spikes + C2_OUT_SIZE;                  // 256
    float* s_fc1_out_spikes = s_p2_out_spikes + P2_OUT_SIZE;                  // 120
    float* s_fc2_out_spikes = s_fc1_out_spikes + FC1_OUT_SIZE;                // 84
    float* s_vm_if1         = s_fc2_out_spikes + FC2_OUT_SIZE;                // 3456
    float* s_vm_if2         = s_vm_if1 + C1_OUT_SIZE;                         // 1024
    float* s_vm_if3         = s_vm_if2 + C2_OUT_SIZE;                         // 120
    float* s_vm_if4         = s_vm_if3 + FC1_OUT_SIZE;                        // 84
    float* s_output_sum     = s_vm_if4 + FC2_OUT_SIZE;                        // 10
    // 总计: 11282 floats (45128 字节)

    // --- 2. 初始化 (t=0) (与 V1 相同) ---
    
    // (协作) 1. 将 1 张图像从全局内存加载到共享内存
    for (int i = tid; i < IMG_SIZE; i += blockDim.x) {
        s_img[i] = d_image[(long long)batch_idx * IMG_SIZE + i];
    }
    
    // (协作) 2. 重置所有 Vm 缓冲区
    for (int i = tid; i < VMS_OUT_SIZE; i += blockDim.x) {
        s_vm_if1[i] = 0.0f; // s_vm_if1 是 Vm 缓冲区的起始地址
    }
    
    if (tid < FC3_OUT_SIZE) s_output_sum[tid] = 0.0f;

    __syncthreads(); // 等待加载和重置完成

    // --- 3. SNN T步循环 (在共享内存中) ---
    for (int t = 0; t < T; ++t) {
        
        // L1: [DENSE] Conv1(s_img) + IF1 -> (s_c1_out_spikes)
        device_conv_bias_if_DENSE(s_img, d_conv1_w, d_conv1_b, s_vm_if1, s_c1_out_spikes,
                                  IMG_C, IMG_H, IMG_W, C1_OUT_C, C1_K);
        __syncthreads();

        // L2: Pool1(s_c1_out_spikes) -> (s_p1_out_spikes)
        device_max_pool2d(s_c1_out_spikes, s_p1_out_spikes,
                          P1_OUT_C, C1_OUT_H, C1_OUT_W, P1_K);
        __syncthreads();
        
        // L3: [SPARSE] Conv2(s_p1_out_spikes) + IF2 -> (s_c2_out_spikes)
        device_conv_bias_if_SPARSE(s_p1_out_spikes, d_conv2_w, d_conv2_b, s_vm_if2, s_c2_out_spikes,
                                   P1_OUT_C, P1_OUT_H, P1_OUT_W, C2_OUT_C, C2_K);
        __syncthreads();

        // L4: Pool2(s_c2_out_spikes) -> (s_p2_out_spikes)
        device_max_pool2d(s_c2_out_spikes, s_p2_out_spikes,
                          P2_OUT_C, C2_OUT_H, C2_OUT_W, P2_K);
        __syncthreads();

        // L5: [SPARSE] FC1(s_p2_out_spikes) + IF3 -> (s_fc1_out_spikes)
        device_linear_bias_if_SPARSE(s_p2_out_spikes, d_fc1_w, d_fc1_b, s_vm_if3, s_fc1_out_spikes,
                                     P2_OUT_SIZE, FC1_OUT_SIZE);
        __syncthreads();

        // L6: [SPARSE] FC2(s_fc1_out_spikes) + IF4 -> (s_fc2_out_spikes)
        device_linear_bias_if_SPARSE(s_fc1_out_spikes, d_fc2_w, d_fc2_b, s_vm_if4, s_fc2_out_spikes,
                                     FC1_OUT_SIZE, FC2_OUT_SIZE);
        __syncthreads();

        // L7: [SPARSE] FC3(s_fc2_out_spikes) -> Accumulate(s_output_sum)
        device_linear_bias_accum_SPARSE(s_fc2_out_spikes, d_fc3_w, d_fc3_b, s_output_sum,
                                        FC2_OUT_SIZE, FC3_OUT_SIZE);
        
        __syncthreads();
    }
    
    // --- 4. 写回结果 (不变) ---
    if (tid < FC3_OUT_SIZE) {
        d_output_sum[(long long)batch_idx * FC3_OUT_SIZE + tid] = s_output_sum[tid];
    }
}


// ===================================================================================
// SCNN Inference Function (Mega-Kernel V1 布局)
// ===================================================================================

std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    // Device pointers for parameters
    float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b,
    float* d_image,
    float* d_output_sum,
    int BATCH_SIZE
    )
{
    std::vector<int> predictions;
    predictions.reserve(BATCH_SIZE);

    const int T = 8; // 时间步长
    const int IMG_SIZE = 784;
    const int FC3_OUT_SIZE = 10;
    
    // --- 1. 展平图像数据 (不变) ---
    std::vector<float> h_all_images( (long long)BATCH_SIZE * IMG_SIZE );
    for (int i = 0; i < BATCH_SIZE; ++i) {
        std::copy(images[i].begin(), images[i].end(), h_all_images.begin() + (long long)i * IMG_SIZE);
    }
    
    // --- 2. 一次性拷贝图像 (不变) ---
    checkCudaErrors(cudaMemcpy(d_image, h_all_images.data(), h_all_images.size() * sizeof(float), cudaMemcpyHostToDevice));

    // --- 3. [V3] 启动 "超级内核 V3" ---
    
    const int THREADS = 512;

    // V1 共享内存大小
    const size_t SHMEM_SIZE = 11282 * sizeof(float); // 45128 字节

    // 启动 BATCH_SIZE (10000) 个块
    scnn_inference_mega_kernel<<<BATCH_SIZE, THREADS, SHMEM_SIZE>>>(
        d_image, d_output_sum,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b,
        T, BATCH_SIZE
    );
    
    // --- 4. 一次性拷贝结果 (不变) ---
    std::vector<float> h_all_outputs( (long long)BATCH_SIZE * FC3_OUT_SIZE );
    checkCudaErrors(cudaMemcpy(h_all_outputs.data(), d_output_sum, h_all_outputs.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // --- 5. CPU ArgMax (不变) ---
    for (int i = 0; i < BATCH_SIZE; ++i) {
        int prediction = 0;
        float max_val = -FLT_MAX;
        float* output_ptr = h_all_outputs.data() + (long long)i * FC3_OUT_SIZE;
        for (int j = 0; j < FC3_OUT_SIZE; ++j) {
            if (output_ptr[j] > max_val) {
                max_val = output_ptr[j];
                prediction = j;
            }
        }
        predictions.push_back(prediction);
    }
    
    return predictions;
}

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
	std::string dir = argv[1];
	
    // Load test data
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");
    
    // --- 1. Allocate all necessary GPU memory (同 V1) ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    // --- Allocate input/output buffers (同 V1) ---
    float *d_image, *d_output_sum;
    const int BATCH_SIZE = images.size(); // 10000
    const int IMG_SIZE = 784;
    const int FC3_OUT_SIZE = 10;
    
    // [FIXED] 修复了拼写错误和重复行
    checkCudaErrors(cudaMalloc(&d_image,       (long long)BATCH_SIZE * IMG_SIZE       * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_output_sum,  (long long)BATCH_SIZE * FC3_OUT_SIZE    * sizeof(float)));
    // --- End of additional allocation ---


    // --- 2. Copy constant parameters from host to device (同 V1) ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // (V1/V3 不需要设置共享内存属性，因为 45kB < 48kB 默认值)

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================

    // --- 3. Perform inference ---
    std::vector<int> predictions = scnn_inference(images,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b,
        d_image,
        d_output_sum,
        BATCH_SIZE
        );
    
// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // --- 4. Free all allocated GPU memory (同 V1) ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    // [FIXED] 修复了拼写错误 d_fc1_c -> d_fc1_w
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));

    // --- Free additional GPU memory ---
    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(d_output_sum));
    
    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();
    
    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    
    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================