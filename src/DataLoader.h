#ifndef DATALOADER_H
#define DATALOADER_H
#include <algorithm>
#include <memory>
#include <vector>
#include <random>

#include "Dataset.h"

// Represents a batch of input and target tensors
struct Batch {
	TensorPtr inputs;
	TensorPtr targets;
};

// DataLoader class that handles batching and shuffling of data
class DataLoader {
public:
	DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle = true);

	class Iterator {
	public:
		Iterator() : loader_(nullptr), position_(0) {} // Default constructor
		Iterator(DataLoader* loader, size_t start_idx): loader_(loader), position_(start_idx) {};

		bool operator!=(const Iterator& other) const {
			// return position_ < loader_->dataset_->size();
			 return position_ < loader_->dataset_->size() - loader_->batch_size_ + 1;
		}

		Iterator& operator++() {
			position_ += loader_->batch_size_;
			return *this;
		}

		Iterator operator++(int) {  // Postfix increment
			Iterator temp = *this;
			position_ += loader_->batch_size_;
			return temp;
    	}

		Batch operator*() const {
			return loader_->getBatch(position_);
		}

	private:

		DataLoader* loader_;
		size_t position_;
	};

	Iterator begin() {
		if(shuffle_) {
			std::random_device rd;
			std::mt19937 g(rd());
			std::shuffle(indices_.begin(), indices_.end(), g);
		}

		return Iterator(this, 0);
	}

	Iterator end() {
		return Iterator(this, dataset_->size());
	}
	size_t totalBatches() const;
	void resetEpoch();
	size_t getCurrentEpoch() const;
	void resetBatchTensors(Batch batch) const;
private:
	Batch getBatch(size_t start_idx);
	static TensorPtr stackTensors(std::vector<TensorPtr>& tensors);

	std::shared_ptr<Dataset> dataset_;
	size_t batch_size_;
	bool shuffle_;
	std::vector<size_t> indices_;
	size_t total_batches_;
	size_t current_epoch_ = 0;
};



#endif //DATALOADER_H
