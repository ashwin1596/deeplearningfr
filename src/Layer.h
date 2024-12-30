//
// Created by ashwi on 11/12/2024.
//

#ifndef LAYER_H
#define LAYER_H
#include "Graph.h"
//#include "json.hpp"


class Layer {
public:
	Layer(int layer_num): layer_num_(layer_num) {}
	virtual NodePtr forward(NodePtr input) = 0;
	virtual std::vector<NodePtr> parameters() = 0;

	//virtual void saveParameters(nlohmann::json& json_data) = 0;/*
	//virtual void loadParameters(const nlohmann::json& json_data) = 0;*/
	virtual std::string getLayerType() = 0;

	virtual ~Layer() = default;

	int layer_num_;
};

// Linear layer implementation
class Linear : public Layer {
public:
	Linear(size_t in_features, size_t out_features, GraphBuilder& builder, int layer_num);

	NodePtr forward(NodePtr input) override;
	std::vector<NodePtr> parameters() override;

	//void saveParameters(nlohmann::json& json_dat/*a) override;

	//void loadParameters(const nlohmann::json &json_data) override;*/
	std::string getLayerType() override;

private:
	GraphBuilder& builder_;
	NodePtr weights_;
	NodePtr bias_;
	size_t in_features_;
	size_t out_features_;
};

// ReLU layer implementation
class ReLU : public Layer {
public:
	ReLU(GraphBuilder& builder, int layer_num);

	NodePtr forward(NodePtr input) override;
	std::vector<NodePtr> parameters() override;
	//void saveParameters(nl/*ohmann::json& json_data) override;

	//void loadParameters(const nlohmann::json &json_data) override;*/
	std::string getLayerType() override;
private:
	GraphBuilder& builder_;
};

// Sequential container for layers
class Sequential {
public:
	Sequential(GraphBuilder& builder);

	void addLayer(std::shared_ptr<Layer> layer);

	NodePtr forward(NodePtr input);

	std::vector<NodePtr> parameters();

	//void saveModel(const std::string& filepath);

	//void loadModel(const std::string& filepath);
private:
	GraphBuilder& builder_;
	std::vector<std::shared_ptr<Layer>> layers_;
};

#endif //LAYER_H
