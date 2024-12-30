//
// Created by ashwi on 11/12/2024.
//

#include "Layer.h"

#include <iostream>
#include <fstream>
#include <bits/random.h>
#include <random>

Sequential::Sequential(GraphBuilder &builder): builder_(builder) {
}

void Sequential::addLayer(std::shared_ptr<Layer> layer) {
	layers_.push_back(layer);
}

NodePtr Sequential::forward(NodePtr input) {
	NodePtr output = input;
	for (auto &layer: layers_) {
		output = layer->forward(output);
	}
	return output;
}

std::vector<NodePtr> Sequential::parameters() {
	std::vector<NodePtr> params;
	for (auto &layer: layers_) {
		auto layer_params = layer->parameters();
		params.insert(params.end(), layer_params.begin(), layer_params.end());
	}
	return params;
}

//void Sequential::saveModel(const std::string &filepath) {
//	nlohmann::json model_data;
//	model_data["num_layers"] = layers_.size();
//
//	//Save each layer
//	for (size_t i = 0; i < layers_.size(); ++i) {
//		nlohmann::json layer_data;
//		layers_[i]->saveParameters(layer_data);
//		model_data["layer_" + std::to_string(i)] = layer_data;
//	}
//
//	//Write to file
//	std::ofstream file(filepath);
//	file << model_data.dump(4);
//}
//
//void Sequential::loadModel(const std::string &filepath) {
//	//Read from file
//	std::ifstream file(filepath);
//	nlohmann::json model_data;
//	file >> model_data;
//
//
//	size_t num_layers = model_data["num_layers"];
//
//	//Create layers
//	for (size_t i = 0; i < num_layers; ++i) {
//		nlohmann::json layer_data = model_data["layer_" + std::to_string(i)];
//		std::string layer_type = layer_data["type"];
//		std::shared_ptr<Layer> layer;
//		if (layer_type == "Linear") {
//			layer = std::make_shared<Linear>(layer_data["in_features"], layer_data["out_features"], builder_, i);
//		} else if (layer_type == "ReLU") {
//			layer = std::make_shared<ReLU>(builder_, i);
//		}
//		layers_.push_back(layer);
//	}
//
//	//Load each layer
//	for (size_t i = 0; i < num_layers; ++i) {
//		nlohmann::json layer_data = model_data["layer_" + std::to_string(i)];
//		layers_[i]->loadParameters(layer_data);
//	}
//}


Linear::Linear(size_t in_features, size_t out_features, GraphBuilder &builder, int layer_num): Layer(layer_num), builder_(builder),
	in_features_(in_features), out_features_(out_features){
	// Initialize weights and bias
	std::random_device rd;

	std::mt19937 gen(rd());

	//TODO : check this initialization
	std::normal_distribution<float> dist(0.0, std::sqrt(2.0 / in_features));

	auto weight_data = std::make_shared<Tensor>(std::vector<size_t>{out_features, in_features});
	auto bias_data = std::make_shared<Tensor>(std::vector<size_t>{out_features});

	for (size_t i = 0; i < out_features; ++i) {
		for (size_t j = 0; j < in_features; ++j) {
			float temp = dist(gen);
			(*weight_data->data)[weight_data->index({i, j})] = temp;
		}
		(*bias_data->data)[i] = dist(gen);
	}

	auto& config = Config::getInstance();
	auto device_ = config.getDeviceType();
	if(device_ == DeviceType::GPU){
		weight_data->copyToGPU();
		bias_data->copyToGPU();
	}

	weights_ = builder.createVariable("weights_"+std::to_string(layer_num_), weight_data);
	bias_ = builder.createVariable("bias_"+std::to_string(layer_num_), bias_data);
}

NodePtr Linear::forward(NodePtr input) {
	auto wx = builder_.createMatmulNode(weights_, input, "wx_"+std::to_string(layer_num_));
	auto wx_plus_b = builder_.createAddNode(wx, bias_, "wx_plus_b_"+std::to_string(layer_num_));
	return wx_plus_b;
}

std::vector<NodePtr> Linear::parameters() {
	return {weights_, bias_};
}

//void Linear::saveParameters(nlohmann::json &json_data) {
//	//Save weights and bias tensors
//	json_data["type"] = "Linear";
//	json_data["in_features"] = in_features_;
//	json_data["out_features"] = out_features_;
//
//	//Convert weights to json array
//	json_data["weights"] = std::vector<float>(weights_->value()->get_data()->begin(), weights_->value()->get_data()->end());
//
//	//Convert bias to json array
//	json_data["bias"] = std::vector<float>(bias_->value()->get_data()->begin(), bias_->value()->get_data()->end());
//}
//
//void Linear::loadParameters(const nlohmann::json &json_data) {
//	//Load weights and bias tensors
//	std::vector<float> weights_data = json_data["weights"];
//	std::vector<float> bias_data = json_data["bias"];
//
//	//Create weights and bias tensors
//	auto weight_tensor = std::make_shared<Tensor>(std::vector<size_t>{out_features_, in_features_}, weights_data);
//	auto bias_tensor = std::make_shared<Tensor>(std::vector<size_t>{out_features_}, bias_data);
//
//	weights_->setValue(weight_tensor);
//	bias_->setValue(bias_tensor);
//}

std::string Linear::getLayerType() {
	return "Linear";
}

// ReLU layer implementation
ReLU::ReLU(GraphBuilder &builder, int layer_num): Layer(layer_num), builder_(builder) {
}

NodePtr ReLU::forward(NodePtr input) {
	auto output = builder_.createReluNode(input, "relu_"+std::to_string(layer_num_));
	return output;
}

std::vector<NodePtr> ReLU::parameters() {
	return {};
}

//void ReLU::saveParameters(nlohmann::json &json_data) {
//	json_data["type"] = "ReLU";
//}
//
//void ReLU::loadParameters(const nlohmann::json &json_data) {
//}

std::string ReLU::getLayerType() {
	return "Relu";
}

