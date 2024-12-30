#include "Dataset.h"

#include <iostream>
#include <fstream>
#include <sstream>


	MNISTDataset::MNISTDataset(const std::string& image_file_path) {
		file_path = image_file_path;
		loadLibSVMData(image_file_path);
	}

	size_t MNISTDataset::size() const{
		return images_.size();
	}

	void MNISTDataset::reload(){
		images_.clear();
		labels_.clear();
		loadLibSVMData(file_path);
	}

	std::pair<TensorPtr, TensorPtr> MNISTDataset::get(size_t idx){
		return std::pair<TensorPtr, TensorPtr>(images_[idx], labels_[idx]);
		// return std::pair<TensorPtr, TensorPtr>(std::move(images_[idx]), std::move(labels_[idx]));
	}

	void MNISTDataset::loadLibSVMData(const std::string& data_file) {
		std::ifstream file(data_file);
		if(!file.is_open()) {
			throw std::runtime_error("Could not open file: " + data_file);
		}

		std::string line;
		const int num_features = 784;
		const int num_labels = 10;

		while(std::getline(file, line)) {
			//Initialize image data with zeros
			std::vector<float> image_data(num_features, 0.0);
			std::vector<float> label_data(num_labels, 0.0);

			//Parse label and features
			std::istringstream iss(line);
			int label;
			iss >> label;

			//Set one-hot encoding for label
			if(label >=0 && label <10) {
				label_data[label] = 1.0;
			}
			else {
				throw std::runtime_error("Invalid label: " + std::to_string(label));
			}

			//Parse features
			std::string feature_pair;
			while(iss >> feature_pair) {
				size_t colon_pos = feature_pair.find(':');
				if(colon_pos == std::string::npos) {
					throw std::runtime_error("Invalid feature pair: " + feature_pair);
				}

				int index = std::stoi(feature_pair.substr(0, colon_pos)) - 1; // 1-indexed LibSVM format
				float value = std::stod(feature_pair.substr(colon_pos + 1));

				if(index >= 0 && index < num_features) {
					image_data[index] = value / 255.0;
				}
				else {
					throw std::runtime_error("Invalid feature index: " + std::to_string(index));
				}
			}

	
			//Store image and label data as 1D tensors
			images_.push_back(std::make_shared<Tensor>(std::vector<size_t>{static_cast<size_t>(num_features)}, image_data));
			labels_.push_back(std::make_shared<Tensor>(std::vector<size_t>{static_cast<size_t>(num_labels)}, label_data));
		}

		if (images_.empty()) {
			throw std::runtime_error("No data loaded from file: " + data_file);
		}

		if(images_.size() != labels_.size()) {
			throw std::runtime_error("Number of images and labels do not match");
		}
	}
