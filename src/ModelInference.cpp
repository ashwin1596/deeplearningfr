//
// Created by ashwi on 11/16/2024.
//

#include "ModelInference.h"

	ModelInference::ModelInference(const std::string& model_path, GraphBuilder& builder): builder_(builder) {
		//Load model architecture and weights
		model_ = std::make_shared<Sequential>(builder);
		model_->loadModel(model_path);
	}

	TensorPtr ModelInference::predict(const TensorPtr& input) {
		// Create input node
		auto input_node = builder_.createVariable("inference_input", input);

		NodePtr logits = model_->forward(input_node);

		// Apply softmax to get probabilities
		// NodePtr probabilities = builder_.createSoftmaxNode(logits, "prediction_softmax");
		NodePtr probabilities;

		// Return probability distribution
		return probabilities->value();
	}

	std::vector<size_t> ModelInference::predictClasses(const TensorPtr& input, const size_t num_classes, const size_t batch_size) {
		// Get probability distribution
		TensorPtr probs = predict(input);

		std::vector<size_t> predicted_classes(batch_size);

		// For each sample in batch
		for(size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
			// Find max probability for this sample
			size_t predicted_class = 0;
			float max_prob = (*probs->data)[batch_idx * num_classes]; // First class prob for this sample

			// Check all classes for this sample
			for(size_t class_idx = 1; class_idx < num_classes; class_idx++) {
				size_t idx = batch_idx * num_classes + class_idx;
				if((*probs->data)[idx] > max_prob) {
					max_prob = (*probs->data)[idx];
					predicted_class = class_idx;
				}
			}

			predicted_classes[batch_idx] = predicted_class;
		}

		// Return vector of predictions
		return predicted_classes;
	}