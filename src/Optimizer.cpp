#include "Optimizer.h"

#include <vector>

#include "Graph.h"

// SGD optimizer implementation
	SGD::SGD(std::vector<NodePtr> parameters, float learning_rate): parameters_(parameters), learning_rate_(learning_rate) {}

	void SGD::step() {
		for(auto& param : parameters_) {
			auto grad = param->adjoint()->get_adjoint();
			auto current_value = param->value();
				
			// std::cout << "current_value: ";
			// current_value->print();
			// std::cout << std::endl;

			// Update parameter
			TensorPtr new_value = current_value->subtract(grad->mult(learning_rate_));

			// std::cout << "new_value: ";
			// new_value->print();
			// std::cout << std::endl;

			param->setValue(new_value);
		}
	}

	void SGD::zero_grad() {
		for(auto& param : parameters_) {
			if(param->adjoint() != nullptr) {
				param->adjoint()->set_adjoint(std::make_shared<Tensor>(*param->value()->dims, 0.0f, true));
			}
		}
	}