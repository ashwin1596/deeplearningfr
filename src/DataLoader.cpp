#include "DataLoader.h"

#include <numeric>

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle):
	dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle) {

	resetEpoch();
}

void DataLoader::resetEpoch(){
	    // Recreate indices for the dataset
        indices_.resize(dataset_->size());
        std::iota(indices_.begin(), indices_.end(), 0);

        // If shuffle is enabled, randomly shuffle the indices
        if (shuffle_) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }

		// dataset_->reload();

        // Recalculate total batches
        total_batches_ = (dataset_->size() + batch_size_ - 1) / batch_size_;
        
        // Increment epoch counter
        current_epoch_++;
}

    size_t DataLoader::getCurrentEpoch() const {
        return current_epoch_;
    }

Batch DataLoader::getBatch(size_t start_idx){
    // Check if we've reached or exceeded the dataset size
    if (start_idx >= dataset_->size()) {
        // Return an invalid/empty batch to signal no more data
        return Batch{nullptr, nullptr};
    }

	size_t actual_batch_size = std::min(batch_size_, dataset_->size() - start_idx);
	//Initialize batch tensors
	std::vector<TensorPtr> batch_inputs;
	std::vector<TensorPtr> batch_targets;
	batch_inputs.reserve(actual_batch_size);
	batch_targets.reserve(actual_batch_size);

	//Collect batch data
	for(size_t i=0; i<actual_batch_size; ++i) {
		auto [input, target] = dataset_->get(indices_[start_idx + i]);

		batch_inputs.push_back(input);
		batch_targets.push_back(target);
	}

	//Stack batch data
	return Batch{
	stackTensors(batch_inputs),
	stackTensors(batch_targets)
	};
}

size_t DataLoader::totalBatches() const {
	return total_batches_;
}

void DataLoader::resetBatchTensors(Batch batch) const{
			batch.inputs->freeGPUMemory();
			batch.targets->freeGPUMemory();
}

TensorPtr DataLoader::stackTensors(std::vector<TensorPtr>& tensors){
	if(tensors.empty()) {
		return std::make_shared<Tensor>();
	}

    // Validate tensors
    for (const auto& tensor : tensors) {
        if (!tensor) {
            throw std::invalid_argument("Null tensor in the input vector.");
        }
    }

	std::vector<size_t> dims = *(tensors[0]->dims); 
	dims.insert(dims.begin(), tensors.size());

	auto ret = std::make_shared<Tensor>(dims);

    // Now stack the tensors element-wise
    for (size_t i = 0; i < tensors.size(); i++) {
        for (size_t j = 0; j < tensors[i]->data->size(); j++) {

            // Using the index method to calculate the correct position in the new tensor
            ret->data->at(ret->index({i, j})) = tensors[i]->data->at(j);  // Use at() for safe indexing

        }
    }

	// Copy the updated CPU data to GPU
    if (ret->device_ == DeviceType::GPU) {
        
        ret->copyToGPU();  // Explicitly copy the stacked data to GPU
    }

	return ret;
}

