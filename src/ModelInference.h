//
// Created by ashwi on 11/16/2024.
//

#ifndef MODELINFERENCE_H
#define MODELINFERENCE_H

#include "Graph.h"
#include "Layer.h"


//Inference wrapper class
class ModelInference {
public:
	ModelInference(const std::string& model_path, GraphBuilder& builder);

	TensorPtr predict(const TensorPtr& input);

	std::vector<size_t> predictClasses(const TensorPtr& input, const size_t num_classes, const size_t batch_size);

private:
	GraphBuilder builder_;
	std::shared_ptr<Sequential> model_;
};



#endif //MODELINFERENCE_H
