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

Linear::Linear(size_t in_features, size_t out_features, GraphBuilder &builder, int layer_num): Layer(layer_num), builder_(builder),
	in_features_(in_features), out_features_(out_features){
	// Initialize weights and bias
	std::random_device rd;

	std::mt19937 gen(rd());

	//TODO : check this initialization
	std::normal_distribution<float> dist(0.0, std::sqrt(2.0 / in_features));

	auto weight_data = std::make_shared<Tensor>(std::vector<size_t>{out_features, in_features}, true);
	auto bias_data = std::make_shared<Tensor>(std::vector<size_t>{out_features}, true);

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

std::string ReLU::getLayerType() {
	return "Relu";
}

