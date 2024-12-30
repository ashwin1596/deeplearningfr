#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

#include "Graph.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"

namespace py = pybind11;

PYBIND11_MODULE(deeplearning, m) {

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<float>())
        .def(py::init<std::vector<size_t>, float>())
        .def(py::init<std::vector<size_t>, std::vector<float>>())
        .def(py::init<std::vector<size_t>>())
		.def("get_data", [](const Tensor& self) {
			// Convert std::vector<float> to Python list
			return py::cast(*self.get_data());
		}, "Get the underlying data as a Python list.")
		.def("get_dims", [](const Tensor& self) {
			// Convert std::vector<size_t> to Python list
			return py::cast(*self.get_dims());
		}, "Get the dimensions of the tensor as a Python list.")
		.def("transpose", &Tensor::transpose)
		.def("print", &Tensor::print);

    py::class_<Config>(m, "Config")
        .def_static("get_instance", &Config::getInstance, py::return_value_policy::reference)
        .def("load_from_env", &Config::loadFromEnv)
        .def("get_device_type", &Config::getDeviceType)
        .def("get_cuda_devices", &Config::getCudaDevices)
        .def("get_batch_size", &Config::getBatchSize)
		.def("get_num_epochs", &Config::getNumEpochs)
        .def("set_device_type", &Config::setDeviceType)
        .def("set_cuda_devices", &Config::setCudaDevices)
        .def("set_batch_size", &Config::setBatchSize)
		.def("set_num_epochs", &Config::setNumEpochs);

	py::enum_<DeviceType>(m, "DeviceType")
		.value("CPU", DeviceType::CPU)
		.value("GPU", DeviceType::GPU)
		.export_values();

	py::class_<DeviceManager>(m, "DeviceManager")
        .def_static("device_type_to_string", &DeviceManager::deviceTypeToString, py::arg("device"),
                    "Convert a DeviceType enum to its string representation.");

	py::class_<ComputationGraph, std::shared_ptr<ComputationGraph>>(m, "ComputationGraph")
	   .def(py::init<>())
	   .def("addNode", &ComputationGraph::addNode)
	   .def("backward", &ComputationGraph::backward);

	py::class_<GraphBuilder, std::shared_ptr<GraphBuilder>>(m, "GraphBuilder")
		.def(py::init<>())
		.def("createVariable", &GraphBuilder::createVariable)
		.def("backward", &GraphBuilder::backward);

	py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
		.def("forward", &Layer::forward, py::arg("input"))
		.def("parameters", &Layer::parameters)
		//.def("save_parameters", &Layer::saveParameters, py::arg("json_data"))
		//.def("load_parameters", &Layer::loadParameters, py::arg("json_data"))
		.def("get_layer_type", &Layer::getLayerType)
		.def_readwrite("layer_num", &Layer::layer_num_);

	py::class_<Linear, Layer, std::shared_ptr<Linear>>(m, "Linear")
		.def(py::init<size_t, size_t, GraphBuilder&, int>(),
			 py::arg("in_features"), py::arg("out_features"), py::arg("builder"), py::arg("layer_num"))
		.def("forward", &Linear::forward, py::arg("input"))
		.def("parameters", &Linear::parameters)
		//.def("save_parameters", &Linear::saveParameters, py::arg("json_data"))
		//.def("load_parameters", &Linear::loadParameters, py::arg("json_data"))
		.def("get_layer_type", &Linear::getLayerType);

	py::class_<ReLU, Layer, std::shared_ptr<ReLU>>(m, "ReLU")
	.def(py::init<GraphBuilder&, int>(), py::arg("builder"), py::arg("layer_num"))
	.def("forward", &ReLU::forward, py::arg("input"))
	.def("parameters", &ReLU::parameters)
	//.def("save_parameters", &ReLU::saveParameters, py::arg("json_data"))
	//.def("load_parameters", &ReLU::loadParameters, py::arg("json_data"))
	.def("get_layer_type", &ReLU::getLayerType);

	py::class_<Sequential, std::shared_ptr<Sequential>>(m, "Sequential")
		.def(py::init<GraphBuilder&>(), py::arg("builder"))
		.def("add_layer", &Sequential::addLayer, py::arg("layer"))
		.def("forward", &Sequential::forward, py::arg("input"))
		.def("parameters", &Sequential::parameters);
	//.def("save_model", &Sequential::saveModel, py::arg("filepath"))
	//.def("load_model", &Sequential::loadModel, py::arg("filepath"));

		py::class_<Node, std::shared_ptr<Node>>(m, "Node")
		.def(py::init<const std::string&, OpType, TensorPtr>())
		.def("value", &Node::value)
		.def("setValue", &Node::setValue)
		.def("name", &Node::name)
		.def("setName", &Node::setName)
		.def("adjoint", &Node::adjoint)
		.def("setAdjoint", &Node::setAdjoint)
		.def("op_type", &Node::op_type);

	py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss")
	.def("forward", &Loss::forward, py::arg("logits"), py::arg("targets"));

	py::class_<CrossEntropyLoss, Loss, std::shared_ptr<CrossEntropyLoss>>(m, "CrossEntropyLoss")
	   .def(py::init<GraphBuilder&>(), py::arg("builder"))
	   .def("forward", &CrossEntropyLoss::forward, py::arg("logits"), py::arg("targets"));

	py::class_<optimizer, std::shared_ptr<optimizer>>(m, "Optimizer")
		.def("step", &optimizer::step, "Performs an optimization step.")
		.def("zero_grad", &optimizer::zero_grad, "Zeros out gradients of all parameters.");

	// SGD class derived from optimizer
	py::class_<SGD, optimizer, std::shared_ptr<SGD>>(m, "SGD")
		.def(py::init<std::vector<NodePtr>, float>(),
			 py::arg("parameters"),
			 py::arg("learning_rate"),
			 "Constructs an SGD optimizer with the given parameters and learning rate.")
		.def("step", &SGD::step, "Performs a single optimization step using SGD.")
		.def("zero_grad", &SGD::zero_grad, "Zeros out gradients of the parameters.");
}