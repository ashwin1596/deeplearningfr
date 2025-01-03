//
// Created by ashwi on 11/20/2024.
//

#ifndef TENSOR_CUH
#define TENSOR_CUH



#include <vector>
#include <stdexcept>
#include <cmath>
#include <string>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Utils.h"

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

// Declare CUDA kernels
__global__ void matmul2d(
	const float* left_data,
	const float* right_data,
	const size_t* left_dims,
	const size_t* right_dims,
	float* output_data,
	const size_t* output_dims
);

__device__ int index(const size_t* dims, const int* coords, int ndims);

class Tensor : public std::enable_shared_from_this<Tensor>{
public:
    std::shared_ptr<std::vector<float>> data;
    std::shared_ptr<std::vector<size_t>> dims;

	// GPU data
    std::shared_ptr<float> d_data = nullptr;
    std::shared_ptr<size_t> d_dims = nullptr;
	size_t total_elements = 1;
	bool gpu_allocated_data = false;
	bool gpu_allocated_dims = false;
	DeviceType device_ = DeviceType::CPU;
	float epsilon = 1e-12;
	bool requires_grad_ = false;


    // Prevent default copy, but allow move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move constructor
    Tensor(Tensor&& other) noexcept {
        data = std::move(other.data);
        dims = std::move(other.dims);
        d_data = other.d_data;
        d_dims = other.d_dims;
        // d_data = std::move(other.d_data);
        // d_dims = std::move(other.d_dims);
        total_elements = other.total_elements;
        gpu_allocated_data = other.gpu_allocated_data;
        gpu_allocated_dims = other.gpu_allocated_dims;
        device_ = other.device_;
		requires_grad_ = other.requires_grad_;

        // Prevent float-free by nulling out the moved object
        other.d_data = nullptr;
        other.d_dims = nullptr;
        other.gpu_allocated_data = false;
        other.gpu_allocated_dims = false;
    }

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            // Free existing GPU memory if allocated
            freeGPUMemory();

            // Move contents
            data = std::move(other.data);
            dims = std::move(other.dims);
			d_data = other.d_data;
			d_dims = other.d_dims;
			// d_data = std::move(other.d_data);
			// d_dims = std::move(other.d_dims);
            total_elements = other.total_elements;
            gpu_allocated_data = other.gpu_allocated_data;
            gpu_allocated_dims = other.gpu_allocated_dims;
            device_ = other.device_;
			requires_grad_ = other.requires_grad_;

            // Prevent float-free
            other.d_data = nullptr;
            other.d_dims = nullptr;
            other.gpu_allocated_data = false;
            other.gpu_allocated_dims = false;
        }
        return *this;
    }

	// Constructors
	Tensor();
	Tensor(float scalar, bool requires_grad = false);
	Tensor(std::vector<size_t> dims, float scalar, bool requires_grad = false);
	Tensor(std::vector<size_t> dims, std::vector<float> data, bool requires_grad = false);
	Tensor(std::vector<size_t> dims, bool requires_grad = false);

    // Destructor
    ~Tensor() {
        if (gpu_allocated_data || gpu_allocated_dims) {
            freeGPUMemory();
        }
    }

	// Allocate and Free GPU memory
	void allocateGPUMemory();
	void freeGPUMemory();

	// sync data bwteen CPU and GPU
	void copyToGPU();
	void copyToCPU();

	// Static methods
	static TensorPtr  ones(std::vector<size_t> dims);

	// Utility methods
	size_t index(std::vector<size_t> x) const;
	void print();
	void print_CPU();
	std::shared_ptr<std::vector<float>> get_data() const;
	std::shared_ptr<std::vector<size_t>> get_dims() const;
	bool requires_grad();
	void set_requires_grad(bool isRequired);

	// Tensor operations
	TensorPtr  reshape(std::vector<size_t> new_dims);

	TensorPtr  transpose() const;
	TensorPtr  transpose_CPU() const;

	TensorPtr  neg_CPU();
	TensorPtr  neg();

	TensorPtr  reciprocal_CPU() const;
	TensorPtr  reciprocal() const;

	TensorPtr  add_CPU(const TensorPtr&  x) const;
	TensorPtr  add(const TensorPtr&  x) const;

	TensorPtr  subtract(const TensorPtr&  x) const;

	TensorPtr  mult_CPU(float x) const;
	TensorPtr  mult(float x) const;

	TensorPtr  elementwise_mult_CPU(const TensorPtr&  x) const;
	TensorPtr  elementwise_mult(const TensorPtr&  x) const;

	TensorPtr  pow_CPU(float x) const;
	TensorPtr  pow(float x) const;

	TensorPtr  relu_CPU() const;
	TensorPtr  relu() const;

	TensorPtr  binarilize_CPU() const;
	TensorPtr  binarilize() const;

	TensorPtr  exp_CPU();
	TensorPtr  exp();

	TensorPtr  matmul_CPU(const TensorPtr & x) const;
	TensorPtr  matmul(const TensorPtr & x) const;

	TensorPtr  ln_CPU() const;
	TensorPtr  ln() const;

	TensorPtr  reduction_sum_CPU(size_t axis) const;
	TensorPtr  reduction_sum(size_t axis) const;

	TensorPtr  softmax_CPU(size_t axis) const;
	TensorPtr  softmax(size_t axis) const;

	TensorPtr  batch_matmul_CPU(const TensorPtr & right) const;
	TensorPtr  batch_matmul(const TensorPtr & right) const;

	TensorPtr  sum_CPU() const;
	TensorPtr  sum() const;

	TensorPtr  divide_CPU(const TensorPtr&  x) const;
	TensorPtr  divide(const TensorPtr&  x) const;

	TensorPtr crossentropy_CPU(const TensorPtr& x) const;
	TensorPtr crossentropy(const TensorPtr& x) const;

	TensorPtr mean_CPU() const;
	TensorPtr mean() const;	
};



#endif //TENSOR_CUH
