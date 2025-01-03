#include<vector>
#include<stdexcept>

#include "Tensor.cuh"
#include "Utils.h"

#define MAX_TENSOR_DIMS 3

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper function for CUDA error checking
inline void checkCuda(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(x) checkCuda(x, __FILE__, __LINE__)

__device__ int index(const size_t* dims, const int* x, const int ndims) {

  int ret = 0;
  int prod = 1;

  for(int i = ndims - 1; i >= 0; --i) {
    if(x[i] >= dims[i]) {
      return -1;
    }
    ret += x[i] * prod;
    prod *= dims[i];
  }
  return ret;
}
__global__ void transpose2d_kernel(const float *input_data, const size_t *input_dims, float *output_data,
                                   const size_t *output_dims) {
  // Calculate thread position
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (row >= input_dims[0] || col >= input_dims[1]) {
    return;
  }

  const int ndims = 2;

  // Prepare coordinate arrays (on stack)
  int out_coords[2], input_coords[2];
  out_coords[0] = col;
  out_coords[1] = row;
  input_coords[0] = row;
  input_coords[1] = col;

  // Write result
  output_data[index(output_dims, out_coords, ndims)] = input_data[index(input_dims, input_coords, ndims)];

    if (!isfinite(output_data[index(output_dims, out_coords, ndims)]))
  printf("transpose2d_kernel\n");
}

__global__ void neg_kernel(float *input_data, float *output_data, size_t total_elements) {
  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (idx >= total_elements) {
    return;
  }

  // Direct array access
  output_data[idx] = -input_data[idx];

    if (!isfinite(output_data[idx]))
  printf("neg_kernel\n");
}
__global__ void ln_kernel(float *input_data, float *output_data, size_t total_elements, float epsilon) {
  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (idx >= total_elements) {
    return;
  }

    // Clip very small values to a minimum epsilon
    float clipped_value = max(input_data[idx], epsilon);
    output_data[idx] = log(clipped_value);
}

__global__ void reciprocal_kernel(float *input_data, float *output_data, size_t total_elements) {
  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (idx >= total_elements) {
    return;
  }

  // Direct array access
  output_data[idx] = 1.0/input_data[idx];
}

__global__ void add_kernel(float *input_data1, float *input_data2, float *output_data, 
                          size_t total_elements, size_t input2_size,
                          bool is_scalar_scalar, bool is_left_scalar, bool is_bias_broadcast) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) {
        return;
    }

    if (is_scalar_scalar) {
        // Only thread 0 needs to do the work for scalar + scalar
        // if (idx == 0) {

            output_data[0] = input_data1[0] + input_data2[0];
            // printf("Scalar-Scalar-> %f + %f = %f", input_data1[0], input_data2[0], output_data[0]);
        // }
    }
    else if (is_left_scalar) {
        // scalar + tensor
        output_data[idx] = input_data2[idx] + input_data1[0];
    }
    else if (is_bias_broadcast) {
        // tensor + bias (broadcasting)
        output_data[idx] = input_data1[idx] + input_data2[idx % input2_size];
    }
    else {
        // Standard element-wise addition
        output_data[idx] = input_data1[idx] + input_data2[idx];
    }

}

__global__ void mult_kernel(float *input_data1, float input_data2, float *output_data, size_t total_elements) {
  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (idx >= total_elements) {
    return;
  }

  // Direct array access
  output_data[idx] = input_data1[idx] * input_data2;
}

__global__ void elementwise_mult_kernel(float *input_data1, float *input_data2, float *output_data,
                                        size_t total_elements, 
                                         bool is_scalar1, bool is_scalar2) {
  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (idx >= total_elements) {
    return;
  }

  // Handle scalar multiplication
  if (is_scalar1 && !is_scalar2) {
    output_data[idx] = input_data1[0] * input_data2[idx];
  } else if (!is_scalar1 && is_scalar2) {
    output_data[idx] = input_data1[idx] * input_data2[0];
  } else if (is_scalar1 && is_scalar2) {
    output_data[idx] = input_data1[0] * input_data2[0];
  } else {
    // Standard element-wise multiplication
    output_data[idx] = input_data1[idx] * input_data2[idx];
  }
}

__global__ void pow_kernel(float *input_data, float *output_data, float power, size_t total_elements) {
  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (idx >= total_elements) {
    return;
  }

  // Direct array access
  output_data[idx] = std::pow(input_data[idx], power);
}

__global__ void relu_kernel(float *input_data, float *output_data, size_t total_elements) {
  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (idx >= total_elements) {
    return;
  }

  // Direct array access
  output_data[idx] = input_data[idx] > 0 ? input_data[idx] : 0;
}

__global__ void binarilize_kernel(float *input_data, float *output_data, size_t total_elements) {
  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (idx >= total_elements) {
    return;
  }

  // Direct array access
  output_data[idx] = input_data[idx] > 0 ? 1 : 0;
}

__global__ void exp_kernel(float *input_data, float *output_data, size_t total_elements) {
  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (idx >= total_elements) {
    return;
  }

  // Direct array access
  output_data[idx] = std::exp(input_data[idx]);
}

// CUDA kernel for reduction sum with optimized indexing
__global__ void reductionSumKernel(
    const float* input,
    float* output,
    const size_t* input_dims,
    const size_t* output_dims,
    size_t num_input_dims,
    size_t reduction_axis,
    size_t total_input_elements,
    size_t total_output_elements
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_output_elements) return;

    // Use local arrays for indexing to leverage device function
    size_t output_idx_array[MAX_TENSOR_DIMS];  // Changed from int to size_t
    size_t input_idx_array[MAX_TENSOR_DIMS];   // Changed from int to size_t

    // Reconstruct multi-dimensional indices
    size_t remaining = tid;
    size_t output_dim_idx = 0;
    for (int dim = num_input_dims - 1; dim >= 0; --dim) {
        if (dim == reduction_axis) continue;

        size_t curr_dim_size = output_dims[output_dim_idx];  // Use output_dims instead
        output_idx_array[dim] = remaining % curr_dim_size;
        remaining /= curr_dim_size;
        output_dim_idx++;
    }

    // Perform reduction sum
    float sum = 0.0;  // Changed from float to float to match input/output type
    for (size_t i = 0; i < input_dims[reduction_axis]; ++i) {
        // Copy output indices and add reduction axis index
        for (size_t dim = 0; dim < num_input_dims; ++dim) {
            input_idx_array[dim] = (dim == reduction_axis) ? i : output_idx_array[dim];
        }

        // Compute flat index directly instead of using separate index function
        size_t flat_idx = 0;
        size_t stride = 1;
        for (int dim = num_input_dims - 1; dim >= 0; --dim) {
            flat_idx += input_idx_array[dim] * stride;
            stride *= input_dims[dim];
        }

        sum += input[flat_idx];
    }

    output[tid] = sum;
}

__global__ void batch_matmul_kernel(
  const float* left,
  const float* right,
  float *result,
  int batch_size,
  int M,
  int K,
  int N
) {
  // 2d grid and block indexing
  int b = blockIdx.x; // batch index
  int i = blockIdx.y; // row index in left
  int j = threadIdx.x; // col index in right

  // Early boundary check
  if (b >= batch_size || i >= M || j >= N) {
    return;
  }

  // perform dot product for this (b,i,j) element
  float sum = 0;
  for (int k = 0; k < K; ++k) {
    sum += left[b * M * K + i * K + k] * right[b * K * N + k * N + j];
  }

  // Write result
  result[b * M * N + i * N + j] = sum;
}

  __global__ void divide_kernel(float* output, float* input, float* divideby, size_t total_element, float epsilon){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float clipped_value = max(divideby[idx], epsilon);

    output[idx] = input[idx] /clipped_value;
  }

  __global__ void divide_scalar_kernel(float* output, float* input, float* divideby, size_t total_element, float epsilon){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= total_element)
      return;
      
    float scalar = divideby[0];
    float clipped_value = max(scalar, epsilon);

    output[idx] = input[idx] / clipped_value;
  }

__global__ void broadcast_mult_kernel(
    float* input_data1, float* input_data2, float* output_data,
    size_t num_samples, size_t num_classes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= num_samples * num_classes) {
    return;
  }
  
  // Calculate sample and class indices
  size_t sample_idx = idx / num_classes;  // Which of the 32 samples
  
  // input_data1 has shape [num_samples, num_classes]
  // input_data2 has shape [num_samples]
  // Broadcast the gradient from input_data2[sample_idx] to all classes
  output_data[idx] = input_data1[idx] * input_data2[sample_idx];
}

Tensor::Tensor() : data(std::make_shared<std::vector<float>>()), dims(std::make_shared<std::vector<size_t>>()){
  total_elements = 0;
  gpu_allocated_data = false;
  gpu_allocated_dims = false;
  requires_grad_ = false;
}

Tensor::Tensor(float scalar, bool requires_grad): data(std::make_shared<std::vector<float>>(1, scalar)), dims(std::make_shared<std::vector<size_t>>()), requires_grad_(requires_grad) {
  total_elements = 1;
    auto& config = Config::getInstance();
  device_ =  config.getDeviceType();
  if(device_ == DeviceType::GPU){
    // Get CUDA configuration
    int cuda_device_ = std::stoi(config.getCudaDevices());
                
    // Set CUDA device before allocation
    cudaSetDevice(cuda_device_);

    // Allocate GPU memory
    allocateGPUMemory();
    copyToGPU();
  }
}

Tensor::Tensor(std::vector<size_t> dims, float scalar, bool requires_grad) : dims(std::make_shared<std::vector<size_t>>(dims)), requires_grad_(requires_grad) {
  total_elements = 1;

  for (const auto &dim: dims) {
    total_elements *= dim;
  }
  data = std::make_shared<std::vector<float>>(total_elements, scalar);

  auto& config = Config::getInstance();
  device_ =  config.getDeviceType();
  if(device_ == DeviceType::GPU){
    // Get CUDA configuration

    int cuda_device_ = std::stoi(config.getCudaDevices());
                
    // Set CUDA device before allocation
    cudaSetDevice(cuda_device_);

    // Allocate GPU memory
    allocateGPUMemory();
    copyToGPU();
  }
}

Tensor::Tensor(std::vector<size_t> dims, bool requires_grad) : dims(std::make_shared<std::vector<size_t>>(dims)), requires_grad_(requires_grad) {
  total_elements = 1;

  for (const auto &dim: dims) {
    total_elements *= dim;
  }

  // Allocate CPU memory
  data = std::make_shared<std::vector<float>>(total_elements);
  auto& config = Config::getInstance();
  device_ =  config.getDeviceType();
  if(device_ == DeviceType::GPU){
    // Get CUDA configuration
    int cuda_device_ = std::stoi(config.getCudaDevices());
                
    // Set CUDA device before allocation
    cudaSetDevice(cuda_device_);

    // Allocate GPU memory
    allocateGPUMemory();
    copyToGPU();
  }
}

Tensor::Tensor(std::vector<size_t> dims, std::vector<float> data, bool requires_grad) : dims(std::make_shared<std::vector<size_t>>(dims)), data(std::make_shared<std::vector<float>>(data)), requires_grad_(requires_grad) {
  // Verify input data
  if (data.empty()) {
    throw std::runtime_error("Input data vector is empty");
  }
  
  total_elements = 1;

  for (const auto &dim: dims) {
    total_elements *= dim;
  }

  // Verify sizes match
  if (data.size() != total_elements) {
    throw std::runtime_error("Data size (" + std::to_string(data.size()) + 
                           ") doesn't match tensor dimensions (" + 
                           std::to_string(total_elements) + ")");
  }

  auto& config = Config::getInstance();
  device_ =  config.getDeviceType();
  if(device_ == DeviceType::GPU){
    // Get CUDA configuration
    int cuda_device_ = std::stoi(config.getCudaDevices());
    
    // Set CUDA device before allocation
    cudaSetDevice(cuda_device_);
    cudaGetLastError();

    // Allocate GPU memory
    allocateGPUMemory();
    copyToGPU();
  }
}

  TensorPtr Tensor::ones(std::vector<size_t> dims){
    TensorPtr ret = std::make_shared<Tensor>(dims);
    for(size_t i = 0;i < ret->data->size();++i)
      (*ret->data)[i] = 1;

    if(ret->device_ == DeviceType::GPU){
      ret->copyToGPU();
    }
    return ret;
  }

void Tensor::allocateGPUMemory() {
  if(gpu_allocated_data && gpu_allocated_dims)
    return;

  if(!gpu_allocated_data && total_elements > 0){
    float *raw_data_ptr = nullptr;    
    CUDA_CHECK(cudaMalloc(&raw_data_ptr, total_elements * sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
    d_data = std::shared_ptr<float>(raw_data_ptr, [](float* ptr) { 
      if(ptr) {cudaFree(ptr);}
      // std::cout<<"Freeing the GPU memory"<<std::endl;} 
    });
    gpu_allocated_data = true;
  }

  if(!gpu_allocated_dims && dims->size() > 0){
    size_t *raw_dims_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&raw_dims_ptr, dims->size() * sizeof(size_t)));
    CUDA_CHECK(cudaGetLastError());
    d_dims = std::shared_ptr<size_t>(raw_dims_ptr, [](size_t* ptr) { 
      if(ptr) cudaFree(ptr); 
    });
    gpu_allocated_dims = true;
  }
}

void Tensor::freeGPUMemory() {
    if (gpu_allocated_data) {
        d_data.reset();  // This will trigger the cudaFree deleter
        gpu_allocated_data = false;
    }
    if (gpu_allocated_dims) {
        d_dims.reset();  // This will trigger the cudaFree deleter
        gpu_allocated_dims = false;
    }
}

void Tensor::copyToGPU() {
  auto& config = Config::getInstance();
  device_ =  config.getDeviceType();

  if(device_ == DeviceType::CPU){
    return;
  }

  if (!data || !dims) {
      throw std::runtime_error("Null or empty data or dimensions");
  }

  // std::cout <<"input-dash: " << data << std::endl;

  if (!gpu_allocated_data || !gpu_allocated_dims) {
    allocateGPUMemory();
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  if(gpu_allocated_data){
    CUDA_CHECK(cudaMemcpy(d_data.get(), data->data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  if(gpu_allocated_dims){
    CUDA_CHECK(cudaMemcpy(d_dims.get(), dims->data(), dims->size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  // std::cout<<"Copied to GPU memory"<<std::endl;

}

void Tensor::copyToCPU() {
  if (!gpu_allocated_data) {
    throw std::runtime_error("No GPU data to copy from");
  }

  if (data == nullptr) {
        data = std::make_shared<std::vector<float>>(total_elements);
  }

  CUDA_CHECK(cudaMemcpy(data->data(), d_data.get(), total_elements * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaGetLastError());
}

size_t Tensor::index(std::vector<size_t> x) const {
  if (x.size() != dims->size())
    throw std::runtime_error("Mismatched dims in index");
  size_t ret = 0;
  size_t prod = 1;
  for (int i = dims->size() - 1; i >= 0; --i) {
    if (x[i] >= (*dims)[i])
      throw std::runtime_error("Index out of bound");
    ret += x[i] * prod;
    prod *= (*dims)[i];
  }
  return ret;
}

TensorPtr Tensor::reshape(std::vector<size_t> new_dims) {
  size_t len = 1;
  for (auto d: new_dims)
    len *= d;
  if (len != data->size())
    throw std::runtime_error("Mismatched dims in reshape");

  auto ret = std::make_shared<Tensor>(new_dims, requires_grad_);

    if (gpu_allocated_data) {
        // Ensure GPU memory is allocated for the new tensor
        ret->allocateGPUMemory();

        // Copy GPU data
        CUDA_CHECK(cudaMemcpy(ret->d_data.get(), d_data.get(), total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    } else {
        // Share the CPU data pointer with the new tensor
        ret->data = data;

        // If on GPU, copy data to GPU for the reshaped tensor
        if (device_ == DeviceType::GPU) {
            ret->copyToGPU();
        }
    }
    return ret;
}

TensorPtr Tensor::transpose_CPU() const {
  if (dims->size() == 2) {
    auto ret = std::make_shared<Tensor>(std::vector<size_t>{(*dims)[1], (*dims)[0]}, requires_grad_);
        for (size_t i = 0; i < (*dims)[0]; ++i) {
            for (size_t j = 0; j < (*dims)[1]; ++j) {
                (*ret->data)[ret->index({j, i})] = (*data)[index({i, j})];
            }
        }
        return ret;
  } else if (dims->size() == 3) {
        auto ret = std::make_shared<Tensor>(std::vector<size_t>{(*dims)[0], (*dims)[2], (*dims)[1]}, requires_grad_);
        for (size_t b = 0; b < (*dims)[0]; ++b) {
            for (size_t i = 0; i < (*dims)[1]; ++i) {
                for (size_t j = 0; j < (*dims)[2]; ++j) {
                    (*ret->data)[ret->index({b, j, i})] = (*data)[index({b, i, j})];
                }
            }
        }
        return ret;
  } else {
    throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
  }
}

TensorPtr Tensor::transpose() const {

  if (device_ == DeviceType::CPU) {
    return transpose_CPU();
  }

  auto ret = std::make_shared<Tensor>(std::vector<size_t>{(*dims)[1], (*dims)[0]}, requires_grad_);

  // Calculate grid and block dimensions
  dim3 threadsPerBlock(16, 16); // Typically 16x16 or 32x32
    dim3 blocksPerGrid(
        ((*dims)[1] + threadsPerBlock.x - 1) / threadsPerBlock.x,
        ((*dims)[0] + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    transpose2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_data.get(), d_dims.get(), ret->d_data.get(), ret->d_dims.get()
    );
  CUDA_CHECK(cudaGetLastError());

  return ret;
}

TensorPtr Tensor::neg_CPU() {
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    for (size_t i = 0; i < data->size(); ++i) {
        (*ret->data)[i] = -(*data)[i];
    }
    return ret;
}

TensorPtr Tensor::neg() {
    if (device_ == DeviceType::CPU) {
        return neg_CPU();
    }

    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);

    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    neg_kernel<<<numBlocks, blockSize>>>(d_data.get(), ret->d_data.get(), total_elements);
    CUDA_CHECK(cudaGetLastError());

    return ret;
}

TensorPtr Tensor::reciprocal_CPU() const {
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    for (size_t i = 0; i < data->size(); ++i) {
        (*ret->data)[i] = 1.0 / (*data)[i];
    }
    return ret;
}


TensorPtr Tensor::reciprocal() const {
    if (device_ == DeviceType::CPU) {
        return reciprocal_CPU();
    }

    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);

    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    reciprocal_kernel<<<numBlocks, blockSize>>>(d_data.get(), ret->d_data.get(), total_elements);
    CUDA_CHECK(cudaGetLastError());

    return ret;
}


TensorPtr Tensor::add_CPU(const TensorPtr& x) const {
    if (dims->empty() && x->dims->empty()) {
        return std::make_shared<Tensor>((*data)[0] + (*(x->data))[0], requires_grad_); // Scalar + scalar
    }
    if (dims->empty()) {
        return x->add(std::const_pointer_cast<Tensor>(shared_from_this())); // Scalar + tensor
        // return x->add(shared_from_this()); // Scalar + tensor
    }
    if (x->dims->empty()) {
        auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
        for (size_t i = 0; i < data->size(); ++i) {
            (*ret->data)[i] = (*data)[i] + (*(x->data))[0];
        }
        return ret;
    }
    if (*dims != *(x->dims) && x->dims->size() == 1 && (*dims)[0] == (*(x->dims))[0]) {
        // Broadcasting for bias
        auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
        for (size_t i = 0; i < data->size(); ++i) {
            (*ret->data)[i] = (*data)[i] + (*(x->data))[i % x->data->size()];
        }
        return ret;
    }
    if (*dims != *(x->dims)) {
        throw std::runtime_error("Mismatched shape in add");
    }

    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    for (size_t i = 0; i < data->size(); ++i) {
        (*ret->data)[i] = (*data)[i] + (*(x->data))[i];
    }
    return ret;
}


TensorPtr Tensor::add(const TensorPtr& x) const {
    if (device_ == DeviceType::CPU) {
        return add_CPU(x);
    }

    // Handle scalar + scalar
    if (dims->empty() && x->dims->empty()) {
        auto ret = std::make_shared<Tensor>(std::vector<size_t>{}, requires_grad_);
        add_kernel<<<1, 1>>>(d_data.get(), x->d_data.get(), ret->d_data.get(),
                             1, 1, true, false, false);
        CUDA_CHECK(cudaGetLastError());
        return ret;
    }

    // Scalar + tensor
    if (dims->empty()) {
        auto ret = std::make_shared<Tensor>(*(x->dims), requires_grad_);
        int blockSize = 256;
        int numBlocks = (x->total_elements + blockSize - 1) / blockSize;
        add_kernel<<<numBlocks, blockSize>>>(d_data.get(), x->d_data.get(), ret->d_data.get(),
                                             x->total_elements, x->total_elements,
                                             false, true, false);
        CUDA_CHECK(cudaGetLastError());
        return ret;
    }

    // Tensor + scalar
    if (x->dims->empty()) {
        auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
        int blockSize = 256;
        int numBlocks = (total_elements + blockSize - 1) / blockSize;
        add_kernel<<<numBlocks, blockSize>>>(d_data.get(), x->d_data.get(), ret->d_data.get(),
                                             total_elements, 1, false, true, false);
        CUDA_CHECK(cudaGetLastError());
        return ret;
    }

    // Bias broadcasting
    if (*dims != *(x->dims) && x->dims->size() == 1 && (*dims)[0] == (*(x->dims))[0]) {
        auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
        int blockSize = 256;
        int numBlocks = (total_elements + blockSize - 1) / blockSize;
        add_kernel<<<numBlocks, blockSize>>>(d_data.get(), x->d_data.get(), ret->d_data.get(),
                                             total_elements, x->total_elements, false, false, true);
        CUDA_CHECK(cudaGetLastError());
        return ret;
    }

    // Mismatched dimensions
    if (*dims != *(x->dims)) {
        throw std::runtime_error("Mismatched shape in add");
    }

    // Element-wise addition
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    add_kernel<<<numBlocks, blockSize>>>(d_data.get(), x->d_data.get(), ret->d_data.get(),
                                         total_elements, total_elements, false, false, false);
    CUDA_CHECK(cudaGetLastError());
    return ret;
}

TensorPtr Tensor::subtract(const TensorPtr& x) const {
    if (*dims != *(x->dims)) {
        throw std::runtime_error("Mismatched shape in subtract");
    }
    return add(x->neg());
}

TensorPtr Tensor::mult_CPU(float x) const {
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    for (size_t i = 0; i < data->size(); ++i) {
        (*ret->data)[i] = (*data)[i] * x;
    }
    return ret;
}


TensorPtr Tensor::mult(float x) const {
    if (device_ == DeviceType::CPU) {
        return mult_CPU(x);
    }

    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);

    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    mult_kernel<<<numBlocks, blockSize>>>(d_data.get(), x, ret->d_data.get(), total_elements);
    CUDA_CHECK(cudaGetLastError());

    return ret;
}


   TensorPtr Tensor::elementwise_mult_CPU(const TensorPtr& x) const {
        // Handle scalar cases first   
        if (dims->empty() || x->dims->empty()) {
            if (dims->empty() && x->dims->empty()) {
                return std::make_shared<Tensor>(data->at(0) * x->data->at(0), requires_grad_);
            }
            const Tensor& non_scalar = dims->empty() ? *x : *this;
            float scalar_val = dims->empty() ? data->at(0) : x->data->at(0);
            
            auto ret = std::make_shared<Tensor>(*non_scalar.dims, requires_grad_);
            for (size_t i = 0; i < non_scalar.data->size(); ++i) {
                ret->data->at(i) = scalar_val * non_scalar.data->at(i);
            }
            return ret;
        }

        // Handle broadcasting case   
        if (*dims != *x->dims) {
            // Check if broadcasting is possible     
            if (dims->size() != 2 || x->dims->size() != 1 ||
                dims->at(1) != x->dims->at(0)) {  // [10,32] [32]
                throw std::runtime_error("Incompatible shapes for broadcasting in elementwise_mult");
            }
            
            // Broadcast x across second dimension (classes)     
            auto ret = std::make_shared<Tensor>(*dims, requires_grad_);  // Shape will be [10, 32]     
            for (size_t i = 0; i < dims->at(1); ++i) {  // For each of the 32 samples       
                for (size_t j = 0; j < dims->at(0); ++j) {  // For each of the 10 classes         
                    ret->data->at(j * dims->at(1) + i) = 
                        data->at(j * dims->at(1) + i) * x->data->at(i);       
                }     
            }     
            return ret;   
        }    

        // Standard elementwise multiplication   
        auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
        for (size_t i = 0; i < data->size(); ++i) {     
            ret->data->at(i) = data->at(i) * x->data->at(i);   
        }   
        return ret; 
    }

 TensorPtr Tensor::elementwise_mult(const TensorPtr& x) const {
        if (device_ == DeviceType::CPU) {
            return elementwise_mult_CPU(x);
        }
        
        bool is_scalar1 = dims->empty();   
        bool is_scalar2 = x->dims->empty();      

        // Handle scalar multiplication   
        if (is_scalar1 || is_scalar2) {     
            if (is_scalar1 && is_scalar2) {       
                auto ret = std::make_shared<Tensor>(1, requires_grad_);
                elementwise_mult_kernel<<<1, 1>>>(       
                d_data.get(), x->d_data.get(), ret->d_data.get(),       
                ret->total_elements,       
                is_scalar1, is_scalar2     
                ); 
                return ret;   
            }     
            const Tensor& non_scalar = is_scalar1 ? *x : *this;     
            auto ret = std::make_shared<Tensor>(*non_scalar.dims, requires_grad_);
            int blockSize = 256;     
            int numBlocks = (ret->total_elements + blockSize - 1) / blockSize;     
            elementwise_mult_kernel<<<numBlocks, blockSize>>>(       
                d_data.get(), x->d_data.get(), ret->d_data.get(),       
                ret->total_elements,       
                is_scalar1, is_scalar2     
            );     
            return ret;   
        }    

        // Handle broadcasting case   
        if (*dims != *x->dims) {     
            if (dims->size() != 2 || x->dims->size() != 1 ||
                dims->at(1) != x->dims->at(0)) {  // [10,32] [32]       
                std::cout << *dims << std::endl;         
                std::cout << *x->dims << std::endl;       
                throw std::runtime_error("Incompatible shapes for broadcasting in elementwise_mult");     
            }          
            
            auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
            int blockSize = 256;     
            int numBlocks = (total_elements + blockSize - 1) / blockSize;     
            broadcast_mult_kernel<<<numBlocks, blockSize>>>(       
                d_data.get(), x->d_data.get(), ret->d_data.get(),       
                dims->at(1), dims->at(0)  // samples(32) and classes(10)      
            );     
            return ret;   
        }    

        // Standard elementwise multiplication   
        auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
        int blockSize = 256;   
        int numBlocks = (total_elements + blockSize - 1) / blockSize;   
        elementwise_mult_kernel<<<numBlocks, blockSize>>>(     
            d_data.get(), x->d_data.get(), ret->d_data.get(),     
            total_elements,     
            false, false   
        );   
        return ret; 
    }

TensorPtr Tensor::pow_CPU(float x) const {
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    for (size_t i = 0; i < data->size(); ++i)
        ret->data->at(i) = std::pow(data->at(i), x);
    return ret;
}

TensorPtr Tensor::pow(float x) const {
    if (device_ == DeviceType::CPU) {
        return pow_CPU(x);
    }

    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    pow_kernel<<<numBlocks, blockSize>>>(
        d_data.get(), ret->d_data.get(), x, total_elements
    );
    return ret;
}

TensorPtr Tensor::relu_CPU() const {
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    for (size_t i = 0; i < data->size(); ++i)
        ret->data->at(i) = data->at(i) > 0 ? data->at(i) : 0;
    return ret;
}

TensorPtr Tensor::relu() const {
    if (device_ == DeviceType::CPU) {
        return relu_CPU();
    }

    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    relu_kernel<<<numBlocks, blockSize>>>(
        d_data.get(), ret->d_data.get(), total_elements
    );

    return ret;
}

TensorPtr Tensor::binarilize_CPU() const {
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    for (size_t i = 0; i < data->size(); ++i)
        ret->data->at(i) = data->at(i) > 0 ? 1 : 0;
    return ret;
}

TensorPtr Tensor::binarilize() const {
    if (device_ == DeviceType::CPU) {
        return binarilize_CPU();
    }

    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    binarilize_kernel<<<numBlocks, blockSize>>>(
        d_data.get(), ret->d_data.get(), total_elements
    );
    return ret;
}

TensorPtr Tensor::exp_CPU() {
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    for (size_t i = 0; i < data->size(); ++i)
        ret->data->at(i) = std::exp(data->at(i));
    return ret;
}

TensorPtr Tensor::exp() {
    if (device_ == DeviceType::CPU) {
        return exp_CPU();
    }

    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    exp_kernel<<<numBlocks, blockSize>>>(
        d_data.get(), ret->d_data.get(), total_elements
    );
    return ret;
}

TensorPtr Tensor::matmul_CPU(const TensorPtr& x) const {
    auto left = std::make_shared<Tensor>(*dims, *data, requires_grad_);
    auto right = std::make_shared<Tensor>(*x->dims, *x->data, requires_grad_);
    
    if (x->dims->size() != 2) {
        throw std::runtime_error("The right operand of matmul must be 2D tensors");
    }
    if (dims->size() != 2 && dims->size() != 3) {
        throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
    }

    std::vector<size_t> ret_dims = {left->dims->at(0), right->dims->at(1)};
    auto ret = std::make_shared<Tensor>(ret_dims, requires_grad_);
    
    for (size_t i = 0; i < left->dims->at(0); ++i) {
        for (size_t j = 0; j < right->dims->at(1); ++j) {
            for (size_t k = 0; k < left->dims->at(1); ++k) {
                ret->data->at(ret->index({i, j})) += 
                    left->data->at(left->index({i, k})) * 
                    right->data->at(right->index({k, j}));
            }
        }
    }
    return ret;
}

__global__ void matmul2d(const float* left_data, const float* right_data, const size_t* left_dims, const size_t* right_dims, float* output_data, const size_t* output_dims){
  // Calculate thread position
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Early boundary check
  if (row >= left_dims[0] || col >= right_dims[1]) {
    return;
  }

  const int ndims = 2;

  // Initialize accumulator
  float sum = 0.0;

  // Prepare coordinate arrays (on stack)
  int out_coords[2], left_coords[2], right_coords[2];
  out_coords[0] = row;
  out_coords[1] = col;
  left_coords[0] = row;
  right_coords[1] = col;

  // Loop over inner dimension
  for (int k = 0; k < left_dims[1]; ++k) {
    left_coords[1] = k;
    right_coords[0] = k;

    sum += left_data[index(left_dims, left_coords, ndims)] *
           right_data[index(right_dims, right_coords, ndims)];
  }

  // Write result
  output_data[index(output_dims, out_coords, ndims)] = sum;
                          if (!isfinite(output_data[index(output_dims, out_coords, ndims)]))
  printf("transpose2d_kernel\n");
}

TensorPtr Tensor::matmul(const TensorPtr& x) const {
    if (device_ == DeviceType::CPU) {
        return matmul_CPU(x);
    }

    // Input validation
    if (x->dims->size() != 2) {
        throw std::runtime_error("The right operand of matmul must be 2D tensors");
    }
    if (dims->size() != 2 && dims->size() != 3) {
        throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
    }
    if (dims->at(dims->size() - 1) != x->dims->at(0)) {
        throw std::runtime_error("Mismatched matmul matrix dimensions");
    }

    std::vector<size_t> ret_dims = {dims->at(0), x->dims->at(1)};
    auto ret = std::make_shared<Tensor>(ret_dims, requires_grad_);

    // Calculate grid and block dimensions
    dim3 threadsPerBlock(16, 16); // Typically 16x16 or 32x32
    dim3 blocksPerGrid(
        (x->dims->at(1) + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (dims->at(0) + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    matmul2d<<<blocksPerGrid, threadsPerBlock>>>(
        d_data.get(), x->d_data.get(),
        d_dims.get(), x->d_dims.get(),
        ret->d_data.get(), ret->d_dims.get()
    );

    return ret;
}

TensorPtr Tensor::ln_CPU() const {
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    for(size_t i = 0; i < data->size(); ++i)
        ret->data->at(i) = std::log(data->at(i));
    return ret;
}

TensorPtr Tensor::ln() const {
    if(device_ == DeviceType::CPU) {
        return ln_CPU();
    }

    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    ln_kernel<<<numBlocks, blockSize>>>(
        d_data.get(), 
        ret->d_data.get(), 
        total_elements,
        epsilon
    );
    return ret;
}

TensorPtr Tensor::reduction_sum_CPU(size_t axis) const {
    if(axis >= dims->size())
        throw std::runtime_error("Invalid axis in reduction_sum");

    // Create new dimensions without the reduction axis
    auto new_dims = std::make_shared<std::vector<size_t>>();
    for(size_t i = 0; i < dims->size(); ++i) {
        if(i != axis)
            new_dims->push_back(dims->at(i));
    }

    auto ret = std::make_shared<Tensor>(*new_dims, requires_grad_);
    std::fill(ret->data->begin(), ret->data->end(), 0); // Initialize result to zeros

    // Calculate total number of elements in the input tensor
    size_t total_elements = 1;
    for(const auto& d : *dims)
        total_elements *= d;

    // Iterate through all elements of the input tensor
    for(size_t i = 0; i < total_elements; ++i) {
        // Convert flat index to multi-dimensional indices
        std::vector<size_t> curr_idx(dims->size());
        size_t temp = i;
        for(int j = dims->size() - 1; j >= 0; --j) {
            curr_idx[j] = temp % dims->at(j);
            temp /= dims->at(j);
        }

        // Create output index by removing the reduction axis
        std::vector<size_t> out_idx;
        for(size_t j = 0; j < dims->size(); ++j) {
            if(j != axis)
                out_idx.push_back(curr_idx[j]);
        }

        // Accumulate sum
        size_t out_flat_idx = ret->index(out_idx);
        ret->data->at(out_flat_idx) += data->at(i);
    }

    return ret;
}

TensorPtr Tensor::reduction_sum(size_t axis) const {
    if (device_ == DeviceType::CPU) {
        return reduction_sum_CPU(axis);
    }

    // Validate axis
    if (axis >= dims->size())
        throw std::runtime_error("Invalid axis in reduction_sum");

    // Create new dimensions without the reduction axis
    auto new_dims = std::make_shared<std::vector<size_t>>();
    for (size_t i = 0; i < dims->size(); ++i) {
        if (i != axis)
            new_dims->push_back(dims->at(i));
    }

    // Allocate output tensor on host and device
    auto ret = std::make_shared<Tensor>(*new_dims, requires_grad_);

    size_t total_output_elements = 1;
    for (const auto& d : *new_dims)
        total_output_elements *= d;

    // Configure grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_output_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    reductionSumKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_data.get(),
        ret->d_data.get(),
        d_dims.get(),
        ret->d_dims.get(),
        dims->size(),
        axis,
        total_elements,
        total_output_elements
    );

    return ret;
}

TensorPtr Tensor::batch_matmul_CPU(const TensorPtr& right) const {
    // left shape: (batch_size x M x K) = (32 x 10 x 10) [Jacobian]
    // right shape: (batch_size x K x N) = (32 x 10 x 1) [reshaped adjoint]
    // result shape: (batch_size x M x N) = (32 x 10 x 1)

    auto left = std::make_shared<Tensor>(*dims, *data, requires_grad_);

    const size_t batch_size = left->dims->at(0);  // 32
    const size_t M = left->dims->at(1);           // 10
    const size_t K = left->dims->at(2);           // 10
    const size_t N = right->dims->at(2);          // 1

    // Initialize result tensor with zeros
    std::vector<size_t> result_dims = {batch_size, M, N};
    auto result = std::make_shared<Tensor>(result_dims, requires_grad_);

    // Perform batch matrix multiplication
    for(size_t b = 0; b < batch_size; ++b) {
        for(size_t i = 0; i < M; ++i) {
            for(size_t j = 0; j < N; ++j) {
                float sum = 0;
                for(size_t k = 0; k < K; ++k) {
                    // left[b,i,k] * right[b,k,j]
                    sum += left->data->at(left->index({b,i,k})) *
                          right->data->at(right->index({b,k,j}));
                }
                result->data->at(result->index({b,i,j})) = sum;
            }
        }
    }

    return result;
}

TensorPtr Tensor::batch_matmul(const TensorPtr& right) const {
    if(device_ == DeviceType::CPU) {
        return batch_matmul_CPU(right);
    }

    const size_t batch_size = dims->at(0);
    const size_t M = dims->at(1);
    const size_t K = dims->at(2);
    const size_t N = right->dims->at(2);

    // Configure grid and block dimensions
    dim3 grid(batch_size, M, 1);
    dim3 block(N, 1, 1);

    // Initialize result tensor with zeros
    std::vector<size_t> result_dims = {batch_size, M, N};
    auto result = std::make_shared<Tensor>(result_dims, requires_grad_);

    batch_matmul_kernel<<<grid, block>>>(
        d_data.get(),
        right->d_data.get(),
        result->d_data.get(),
        batch_size, M, K, N
    );

    return result;
}


TensorPtr Tensor::sum_CPU() const {
    auto ret = std::make_shared<Tensor>(0, requires_grad_);
    for(size_t i = 0; i < data->size(); ++i) {
        ret->data->at(0) += data->at(i);
    }
    return ret;
}

TensorPtr Tensor::sum() const {
    if(device_ == DeviceType::CPU) {
        return sum_CPU();
    }
    return reduction_sum(0);
}

TensorPtr Tensor::divide_CPU(const TensorPtr& x) const {
    // Check if shapes are compatible
    if(*dims != *(x->dims) && x->dims->size() != 0) {
        throw std::runtime_error("Incompatible shapes for division:" +
                               std::to_string(dims->size()) + " and " +
                               std::to_string(x->dims->size()));
    }

    // Create result tensor with same shape as this tensor
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);

    // Handle scalar division - broadcasting
    if(x->dims->size() == 0) {
        float scalar_val = x->data->at(0);

        // Divide each element by the scalar
        for(size_t i = 0; i < total_elements; ++i) {
            if(scalar_val == 0) {
                throw std::runtime_error("Divide by zero");
            }
            ret->data->at(i) = data->at(i) / scalar_val;
        }

        return ret;
    }

    // Handle element-wise division
    for(size_t i = 0; i < total_elements; ++i) {
        if(x->data->at(i) == 0) {
            throw std::runtime_error("Divide by zero");
        }
        ret->data->at(i) = data->at(i) / x->data->at(i);
    }

    return ret;
}

TensorPtr Tensor::crossentropy_CPU(const TensorPtr& x) const {
    // Ensure the input tensor is valid
    if (!x || !x->dims || x->dims->size() < 2) {
        throw std::invalid_argument("Invalid input tensor for cross-entropy.");
    }

    // Get the batch size and number of classes
    size_t batch_size = (*x->dims)[0];
    size_t num_classes = (*x->dims)[1];

    // Validate dimensions
    if ((*dims)[0] != batch_size || (*dims)[1] != num_classes) {
        throw std::invalid_argument("Input tensor dimensions do not match target dimensions.");
    }

    // Create the result tensor to hold the cross-entropy loss for each batch
    std::vector<size_t> ret_dims = {batch_size};
    auto ret = std::make_shared<Tensor>(ret_dims, requires_grad_);

    // Iterate over each batch
    for (size_t i = 0; i < batch_size; ++i) {
        // Base pointers for logits and targets
        const float* logits_batch = data->data() + i * num_classes;
        const float* targets_batch = x->data->data() + i * num_classes;

        // Compute the max value for numerical stability (log-sum-exp trick)
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < num_classes; ++j) {
            max_val = std::max(max_val, logits_batch[j]);
        }

        // Compute the sum of exponentials of the logits
        float sum = 0.0;
        for (size_t j = 0; j < num_classes; ++j) {
            sum += std::exp(logits_batch[j] - max_val);
        }

        // Compute the log-sum-exp result
        float log_sum = max_val + std::log(sum);

        // Calculate the cross-entropy loss for this batch
        float loss = 0.0;
        for (size_t j = 0; j < num_classes; ++j) {
            if (targets_batch[j] > 0) { // Only consider non-zero targets (one-hot encoding)
                loss -= targets_batch[j] * (logits_batch[j] - log_sum);
            }
        }

        // Store the computed loss in the result tensor
        ret->data->at(i) = loss;
    }

    // Return the tensor containing the cross-entropy loss for each batch
    return ret;
}

__global__ void cross_entropy_kernel(float* logits, float* targets, float* losses, 
                                   size_t batch_size, size_t num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        // Adjust memory access pattern to match CPU version
        float* logits_batch = logits + idx*num_classes;  // Use striding
        float* targets_batch = targets + idx*num_classes;
        float& loss = losses[idx];

        // Find max for stability
        float max_val = -INFINITY;
        for (size_t i = 0; i < num_classes; ++i) {
            max_val = fmaxf(max_val, logits_batch[i]);  // Stride by batch_size
        }

        // Sum of exponentials
        float sum = 0.0;
        for (size_t i = 0; i < num_classes; ++i) {
            sum += expf(logits_batch[i] - max_val);  // Stride by batch_size
        }

        float log_sum = max_val + logf(sum);

        // Calculate loss
        float loss_val = 0.0;
        for (size_t i = 0; i < num_classes; ++i) {
            if (targets_batch[i] > 0) {  // Add zero check like CPU version
                loss_val -= targets_batch[i] * 
                           (logits_batch[i] - log_sum);
            }
        }

        loss = loss_val;
    }
}

TensorPtr Tensor::crossentropy(const TensorPtr& x) const {
    if(device_ == DeviceType::CPU) {
        return crossentropy_CPU(x);
    }

    size_t batch_size = (*x->dims)[0]; // Batch size (first dimension of x)
    size_t num_classes = (*x->dims)[1]; // Number of classes (second dimension of x)
    std::vector<size_t> ret_dims = {batch_size};  // Shape [batch_size]
    auto ret = std::make_shared<Tensor>(ret_dims, requires_grad_);  // Same shape as batch_size

    size_t block_size = 256;
    size_t num_blocks = (batch_size + block_size - 1) / block_size;

    cross_entropy_kernel<<<num_blocks, block_size>>>(d_data.get(), x->d_data.get(), ret->d_data.get(), batch_size,
                                                    num_classes);

    // cudaDeviceSynchronize();

    return ret;
}

TensorPtr Tensor::mean_CPU() const {
    // Check if tensor is empty
    if (total_elements == 0) {
        throw std::runtime_error("Tensor is empty, cannot compute mean.");
    }

    // Calculate the sum of all elements in the tensor
    float sum = 0.0;
    for (size_t i = 0; i < total_elements; ++i) {
        sum += (*data)[i];  // Assuming `data` is a vector of values
    }

    // Compute the mean
    float mean_val = sum / total_elements;

    // Create a new tensor to store the result (mean is a scalar)
    std::vector<size_t> mean_dims = {1};  // Mean is a scalar, so the tensor is of size 1
    auto mean_tensor = std::make_shared<Tensor>(mean_dims, requires_grad_);
    
    // Set the mean value in the new tensor
    (*mean_tensor->data)[0] = mean_val;

    return mean_tensor;
}

TensorPtr Tensor::softmax_CPU(size_t axis) const {
    // Ensure the axis is valid
    if (axis >= dims->size()) {
        throw std::invalid_argument("Invalid axis for softmax.");
    }

    // Create a tensor to store the softmax result
    auto result = std::make_shared<Tensor>(*dims, requires_grad_);

    // Get the dimensions and data
    const std::vector<size_t>& tensor_dims = *dims;
    size_t axis_size = tensor_dims[axis];
    size_t total_elements = this->total_elements;

    // Calculate the stride and repeat size for the axis
    size_t stride = 1;
    for (int i = axis + 1; i < static_cast<int>(tensor_dims.size()); ++i) {
        stride *= tensor_dims[i];
    }

    size_t repeat = total_elements / (axis_size * stride);

    // Iterate through the tensor to compute softmax along the specified axis
    for (size_t r = 0; r < repeat; ++r) {
        for (size_t s = 0; s < stride; ++s) {
            // Compute the base index for this slice
            size_t base_idx = r * axis_size * stride + s;

            // Step 1: Compute max for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < axis_size; ++i) {
                size_t idx = base_idx + i * stride;
                max_val = std::max(max_val, data->at(idx));
            }

            // Step 2: Compute the sum of exponentials
            float sum_exp = 0.0;
            for (size_t i = 0; i < axis_size; ++i) {
                size_t idx = base_idx + i * stride;
                sum_exp += std::exp(data->at(idx) - max_val); // Stabilized exponential
            }

            // Step 3: Compute softmax values
            for (size_t i = 0; i < axis_size; ++i) {
                size_t idx = base_idx + i * stride;
                result->data->at(idx) = std::exp(data->at(idx) - max_val) / sum_exp;
            }
        }
    }

    return result;
}

__global__ void softmaxKernel(const float* input, float* output, 
                             const int axis_size, const int stride, 
                             const int repeat, const int total_elements) {
    // Calculate which slice this thread handles
    int slice_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_slices = repeat * stride;
    
    if (slice_idx >= num_slices) return;
    
    int r = slice_idx / stride;
    int s = slice_idx % stride;
    
    // Base index for this slice
    int base_idx = r * axis_size * stride + s;
    
    // Step 1: Find max value
    float max_val = -INFINITY;
    for (int i = 0; i < axis_size; ++i) {
        max_val = fmax(max_val, input[base_idx + i * stride]);
    }
    
    // Step 2: Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < axis_size; ++i) {
        sum_exp += expf(input[base_idx + i * stride] - max_val);
    }
    
    // Step 3: Compute softmax values
    for (int i = 0; i < axis_size; ++i) {
        output[base_idx + i * stride] = 
            expf(input[base_idx + i * stride] - max_val) / sum_exp;
    }
}

TensorPtr Tensor::softmax(size_t axis) const {
    if(device_ == DeviceType::CPU) {
        return softmax_CPU(axis);
    }

    // Ensure the axis is valid
    if (axis >= dims->size()) {
        throw std::invalid_argument("Invalid axis for softmax.");
    }

    // Create a tensor to store the softmax result
    auto result = std::make_shared<Tensor>(*dims, requires_grad_);

    // Get dimensions and data
    const std::vector<size_t>& tensor_dims = *dims;
    size_t axis_size = tensor_dims[axis];

    // Calculate the stride and repeat size for the axis
    size_t stride = 1;
    for (int i = axis + 1; i < static_cast<int>(tensor_dims.size()); ++i) {
        stride *= tensor_dims[i];
    }
    size_t repeat = total_elements / (axis_size * stride);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    softmaxKernel<<<gridSize, blockSize>>>(d_data.get(), result->d_data.get(), axis_size, stride, repeat, total_elements);

    return result;
}

__global__ void mean_kernel(const float* data, float* result, size_t total_elements) {
    extern __shared__ float shared_data[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    float sum = 0.0;
    while (idx < total_elements) {
        sum += data[idx];
        idx += gridDim.x * blockDim.x;
    }
    shared_data[tid] = sum;
    
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // Write the result and divide by total elements
    if (tid == 0) {
        atomicAdd(result, shared_data[0] / total_elements);
    }
}

TensorPtr Tensor::mean() const {
    if(device_ == DeviceType::CPU) {
        return mean_CPU();
    }

   auto ret = std::make_shared<Tensor>(0.0, requires_grad_);

   // Determine block size and grid size for kernel launch
    int block_size = 256;  // Block size, can be tuned based on the hardware
    int grid_size = (total_elements + block_size - 1) / block_size;  // Grid size to cover all elements

    // Launch the kernel
    mean_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(d_data.get(), ret->d_data.get(), total_elements);  

    return ret;
}	

TensorPtr Tensor::divide(const TensorPtr& x) const {
    if(device_ == DeviceType::CPU) {
        return divide_CPU(x);
    }

    // Check if shapes are compatible
    if(*dims != *(x->dims) && x->dims->size() != 0) {
        throw std::runtime_error("Incompatible shapes for division:" + 
                                std::to_string(dims->size()) + " and "+
                                std::to_string(x->dims->size()));
    }

    // Create result tensor
    auto ret = std::make_shared<Tensor>(*dims, requires_grad_);

    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    if(x->dims->size() == 0) {
        // Scalar division
        divide_scalar_kernel<<<numBlocks, blockSize>>>(
            ret->d_data.get(), d_data.get(), x->d_data.get(), total_elements, epsilon
        );
    }
    else {
        divide_kernel<<<numBlocks, blockSize>>>(
            ret->d_data.get(), d_data.get(), x->d_data.get(), total_elements, epsilon
        );
    }

    return ret;
}

__global__ void print_kernel(const float* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // if((data[idx] > 0 || data[idx] < 0) && idx==0)
        // if((data[idx] > 0 || data[idx] < 0))
          printf("data[%d] = %f\n", idx, data[idx]);
    }
}

void Tensor::print_CPU() {
    for(const auto& x : *data) {
        printf("%s\n", std::to_string(x).c_str());
    }
}

void Tensor::print(){
    std::cout << "ReachedPrint" << std::endl;
    if(device_ == DeviceType::CPU) {
        return print_CPU();
    }

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    print_kernel<<<numBlocks, blockSize>>>(d_data.get(), total_elements);

      CUDA_CHECK(cudaDeviceSynchronize());
}

std::shared_ptr<std::vector<float>> Tensor::get_data() const {
    return data;
}

std::shared_ptr<std::vector<size_t>> Tensor::get_dims() const {
    return dims;
}

bool Tensor::requires_grad() {
    return requires_grad_;
}

void Tensor::set_requires_grad(bool isRequired) {
    requires_grad_ = isRequired;
}