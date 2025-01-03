#include <queue>
#include <unordered_map>

#include "Graph.h"
#include "Utils.h"
#include "GradMode.hpp"

std::string opTypeToString(OpType op_type) {
	switch (op_type) {
		case OpType::MATMUL: return "MATMUL";
		case OpType::ADD: return "ADD";
		case OpType::SUBTRACT: return "SUBTRACT";
		case OpType::ELEMENTWISE_MULTIPLY: return "ELEMENTWISE_MULTIPLY";
		case OpType::RELU: return "RELU";
		case OpType::LN: return "LN";
		case OpType::SIN: return "SIN";
		case OpType::VARIABLE: return "VARIABLE";
		case OpType::SUM: return "SUM";
		case OpType::REDUCED_SUM: return "REDUCED_SUM";
		case OpType::TRANSPOSE: return "TRANSPOSE";
		case OpType::DIVIDE: return "DIVIDE";
		case OpType::CROSSENTROPY: return "CROSSENTROPY";
		default: return "UNKNOWN";
	}
}

Node::Node(const std::string &name, OpType op_type, TensorPtr value) : name_(name), 
                                                                           value_(std::move(value)), adjoint_(nullptr), op_type_(op_type){
}

const TensorPtr &Node::value() const { return value_; }
void Node::setValue(const TensorPtr &value) { value_ = value; }
const std::string &Node::name() const { return name_; }
void Node::setName(const std::string &name) { name_ = name; }
void Node::setAdjoint(const AdjointNodePtr &adjoint) { adjoint_ = adjoint; }
OpType Node::op_type() const { return op_type_; }
AdjointNodePtr Node::adjoint() const { return adjoint_; }

AdjointNode::AdjointNode(const std::string &name, OpType op_type, NodePtr primal): name_(name), primal_(primal), op_type_(op_type),
	 needs_reduction_(false), adjoint_value_(std::make_shared<Tensor>(0.0f)){
}

void AdjointNode::addDependency(AdjointNodePtr dep, const TensorPtr &partial_derivative) {
	dependencies_.push_back({dep, partial_derivative});

	// Add reverse edge for topological sort
	dep->parents_.insert(this);
}

void AdjointNode::requires_red_sum(bool needs_reduction) { needs_reduction_ = needs_reduction; }
bool AdjointNode::needs_red_sum() const { return needs_reduction_; }

// Process this node's gradient using stored partial derivatives
void AdjointNode::processGradient() {
	// std::cout << "\nProcessing Gradient for Node: " << name_ << " (Op: " << opTypeToString(op_type_) << ")" <<
	// 		std::endl;
	// std::cout << "Upstream gradient (adjoint_value_) shape: " << adjoint_value_->dims << std::endl;
	// std::cout << "adjoint_value: ";
	// adjoint_value_->print();
	// std::cout << std::endl;


    for (const auto &[dep, partial] : dependencies_) {
		// std::cout << "\nProcessing dependency: " << dep->name() << std::endl;
		// std::cout << "Current accumulated gradient shape: " << dep->adjoint_value_->dims << std::endl;
		// std::cout << "Partial derivative shape: " << partial->dims << std::endl;
		// std::cout << "Partial derivative value: ";
		// partial->print();
		// std::cout << std::endl;


        TensorPtr result;
        TensorPtr result_re;
        TensorPtr result_sf;
        
        TensorPtr transposed_adjoint;
        TensorPtr reshaped_adjoint;

        switch (op_type_) {
            case OpType::MATMUL:
                if (adjoint_value_->dims->size() == 0) {
					// std::cout << "Scalar adjoint case (softmax)" << std::endl;
                    result = partial->elementwise_mult(adjoint_value_);
                    dep->adjoint_value_ = dep->adjoint_value_->add(result);
                } else if (dep->name().find("left") != std::string::npos) {
					// std::cout << "Left matrix gradient case" << std::endl;
                    result = adjoint_value_->matmul(partial);
                    dep->adjoint_value_ = dep->adjoint_value_->add(result);
                } else {
					// std::cout << "Right matrix gradient case" << std::endl;
                    result = partial->matmul(adjoint_value_);
                    dep->adjoint_value_ = dep->adjoint_value_->add(result);
                }

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

            case OpType::ADD:
            case OpType::SUM:
				// std::cout << "Add/Sum gradient case" << std::endl;
                result = adjoint_value_->elementwise_mult(partial);

                if (dep->needs_red_sum()) {

                    result = result->reduction_sum(1);

                }

				// std::cout << "partial: ";
				// partial->print(); 
                // std::cout << std::endl;

				// std::cout << "partial: ";
				// partial->print(); 
                //   std::cout << std::endl;

                dep->adjoint_value_ = dep->adjoint_value_->add(result);

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

			case OpType::CROSSENTROPY:
                // Handle cross-entropy gradients
                result = adjoint_value_->elementwise_mult(partial);
                dep->adjoint_value_ = dep->adjoint_value_->add(result);
                break;
            
            case OpType::SUBTRACT:
				// std::cout << "Subtract gradient case" << std::endl;
                dep->adjoint_value_ = dep->adjoint_value_->add(adjoint_value_->elementwise_mult(partial));

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

            case OpType::ELEMENTWISE_MULTIPLY:
				// std::cout << "Elementwise multiplication gradient case" << std::endl;
                result = adjoint_value_->elementwise_mult(partial);
                dep->adjoint_value_ = dep->adjoint_value_->add(result);

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

            case OpType::RELU:
				// std::cout << "Relu gradient case" << std::endl;
                result_re = adjoint_value_->elementwise_mult(partial);
                dep->adjoint_value_ = dep->adjoint_value_->add(result_re);

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

            case OpType::LN:
				// std::cout << "Ln gradient case" << std::endl;
                dep->adjoint_value_ = dep->adjoint_value_->add(
                    adjoint_value_->elementwise_mult(partial)
                );

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

            case OpType::SIN:
                dep->adjoint_value_ = dep->adjoint_value_->add(
                    adjoint_value_->elementwise_mult(partial)
                );

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

            case OpType::DIVIDE:
				// std::cout << "Divide gradient case" << std::endl;
				
				// std::cout << "Before - partial: ";
				// partial->print(); 
                // std::cout << std::endl;

                result = adjoint_value_->elementwise_mult(partial);
                dep->adjoint_value_ = dep->adjoint_value_->add(result);

				// std::cout << "After gradient computation for " << dep->name() 
                //   << ", result: " << result->data 
                //   << std::endl;

				// std::cout << "After - partial: ";
				// partial->print(); 
                // std::cout << std::endl;

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

            case OpType::TRANSPOSE:
				// std::cout << "Transpose gradient case" << std::endl;
                result = adjoint_value_->transpose();
                dep->adjoint_value_ = dep->adjoint_value_->add(result);

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

            case OpType::REDUCED_SUM:
				// std::cout << "Reduced sum gradient case" << std::endl;
                result = partial->elementwise_mult(adjoint_value_);
                dep->adjoint_value_ = dep->adjoint_value_->add(result);

				//  std::cout << "After gradient computation for " << dep->name() 
                //   << ", adjoint_value_: ";
				//   dep->adjoint_value_->print(); 
                //   std::cout << std::endl;

                break;

            default:
                throw std::runtime_error("Unsupported operation");
        }
    }
}

TensorPtr &AdjointNode::get_adjoint() { return adjoint_value_; }

void AdjointNode::set_adjoint(const TensorPtr &tensor) {
	adjoint_value_ = tensor;
}

const std::string &AdjointNode::name() const { return name_; }
void AdjointNode::setName(const std::string &name) { name_ = name; }
OpType AdjointNode::op_type() const { return op_type_; }

// Graph manager that handles topological sorting and backward pass(autodiff)
void ComputationGraph::addNode(AdjointNodePtr node) {
	nodes_.push_back(node);
}

// Performs topological sort and computes gradients
void ComputationGraph::backward(AdjointNodePtr start_node) {
	// Reset all gradients
	for (auto node: nodes_) {
		node->get_adjoint() = std::make_shared<Tensor>(*node->get_adjoint()->dims, 0.0f);
	}

	// Set the gradient of the start node to 1
	start_node->get_adjoint() = std::make_shared<Tensor>(*start_node->get_adjoint()->dims, 1.0f);

	// Get topological order
	auto sorted_nodes = topologicalSort(start_node);

	// Process gradients in reverse topological order
	for (auto node: sorted_nodes) {
		node->processGradient();
	}
}


//topological sort
std::vector<AdjointNodePtr> ComputationGraph::topologicalSort(AdjointNodePtr start_node) {
	std::vector<AdjointNodePtr> result;
	std::queue<AdjointNodePtr> zero_indegree;
	std::unordered_map<AdjointNodePtr, size_t> indegree;

	// Initialize indegree
	for (auto node: nodes_) {
		indegree[node] = node->parents_.size();
		if (indegree[node] == 0) {
			zero_indegree.push(node);
		}
	}

	// Process nodes
	while (!zero_indegree.empty()) {
		auto node = zero_indegree.front();
		zero_indegree.pop();
		result.push_back(node);

		for (const auto &[dep, _]: node->dependencies_) {
			indegree[dep]--;

			if (indegree[dep] == 0) {
				zero_indegree.push(dep);
			}
		}
	}

	return result;
}


GraphBuilder::GraphBuilder() : graph_(std::make_shared<ComputationGraph>()) {
}


NodePtr GraphBuilder::createVariable(const std::string& name, const TensorPtr& value) {
	auto node = std::make_shared<Node>(name, OpType::VARIABLE, value);

	if (value->requires_grad() && GradMode::isEnabled()) {
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::VARIABLE, node);
		node->setAdjoint(adjoint);

		if (name.find("bias") != std::string::npos) {
			adjoint->requires_red_sum(true);
		}

		graph_->addNode(adjoint);
	}
	return node;
}

NodePtr GraphBuilder::createMatmulNode(NodePtr a, NodePtr b, const std::string& name) {
	// Create forward node
	auto node = std::make_shared<Node>(name, OpType::MATMUL);
	TensorPtr result = a->value()->matmul(b->value());

	bool needs_grad = GradMode::isEnabled() && (a->value()->requires_grad() || b->value()->requires_grad());

	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {

		// Create adjoint node
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::MATMUL, node);
		node->setAdjoint(adjoint);
		
		if (a->value()->requires_grad()) {
			// For gradient of A: dL/dC @ B^T, where C = A @ B and @ is matrix multiplication
			adjoint->addDependency(a->adjoint(), b->value()->transpose());
			a->adjoint()->setName(a->name() + "_left");
		}

		if (b->value()->requires_grad()) {
			// For gradient of B: A^T @ dL/dC, where C = A @ B and @ is matrix multiplication
			adjoint->addDependency(b->adjoint(), a->value()->transpose());
			b->adjoint()->setName(b->name() + "_right");
		}

		graph_->addNode(adjoint);
	}
	return node;
}

NodePtr GraphBuilder::createAddNode(NodePtr a, NodePtr b, const std::string& name) {
	auto node = std::make_shared<Node>(name, OpType::ADD);
	TensorPtr result = a->value()->add(b->value());
	bool needs_grad = GradMode::isEnabled() && (a->value()->requires_grad() || b->value()->requires_grad());
	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::ADD, node);
		node->setAdjoint(adjoint);

		if (a->value()->requires_grad()) {
			adjoint->addDependency(a->adjoint(), std::make_shared<Tensor>(1.0f, true));
		}

		if (b->value()->requires_grad()) {
			adjoint->addDependency(b->adjoint(), std::make_shared<Tensor>(1.0f, true));
		}

		graph_->addNode(adjoint);
	}

	return node;
}

NodePtr GraphBuilder::createCrossEntropyNode(NodePtr a, NodePtr b, const std::string& name) {
	// Create forward node
	auto node = std::make_shared<Node>(name, OpType::CROSSENTROPY);
	TensorPtr result = a->value()->transpose()->crossentropy(b->value()->transpose());

	bool needs_grad = GradMode::isEnabled() && (a->value()->requires_grad() || b->value()->requires_grad());
	result->set_requires_grad(needs_grad);

	node->setValue(result);

	if (needs_grad) {
		// Create adjoint node
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::CROSSENTROPY, node);
		node->setAdjoint(adjoint);

		// Compute softmax of 'a' for gradient calculation
		TensorPtr softmax = a->value()->softmax(0);
		// std::cout<<"softmax---"<<std::endl;
		// softmax->print();

		// Partial derivatives
		TensorPtr grad_a = softmax->subtract(b->value()); // ∂L/∂a = softmax(a) - b
		TensorPtr grad_b = softmax->ln()->mult(-1);    // ∂L/∂b = -log(softmax(a))

		// Add dependencies for adjoints
		if (a->value()->requires_grad()) {
			adjoint->addDependency(a->adjoint(), grad_a);  // Attach gradient w.r.t. 'a'
		}

		if (b->value()->requires_grad()) {
			adjoint->addDependency(b->adjoint(), grad_b);  // Attach gradient w.r.t. 'b'
		}

		// Add adjoint to the computational graph
		graph_->addNode(adjoint);
	}
	return node;
}


NodePtr GraphBuilder::createLnNode(NodePtr a, const std::string& name) {
	auto node = std::make_shared<Node>(name, OpType::LN);
	TensorPtr result = a->value()->ln();
	bool needs_grad = GradMode::isEnabled() && a->value()->requires_grad();
	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::LN, node);
		node->setAdjoint(adjoint);

		// std::cout<<"LN partial derivative";
		// a->value()->reciprocal()->print();

		adjoint->addDependency(a->adjoint(), a->value()->reciprocal());

		graph_->addNode(adjoint);
	}
	return node;
}

NodePtr GraphBuilder::createSubtractNode(NodePtr a, NodePtr b, const std::string& name) {
	auto node = std::make_shared<Node>(name, OpType::SUBTRACT);
	TensorPtr result = a->value()->subtract(b->value());
	bool needs_grad = GradMode::isEnabled() && (a->value()->requires_grad() || b->value()->requires_grad());
	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::SUBTRACT, node);
		node->setAdjoint(adjoint);

		// Add dependencies for adjoints
		if (a->value()->requires_grad()) {
			adjoint->addDependency(a->adjoint(), std::make_shared<Tensor>(1.0f, true));
		}

		if (b->value()->requires_grad()) {
			adjoint->addDependency(b->adjoint(), std::make_shared<Tensor>(-1.0f, true));
		}

		graph_->addNode(adjoint);
	}
	return node;
}

NodePtr GraphBuilder::createReluNode(NodePtr a, const std::string& name) {
	auto node = std::make_shared<Node>(name, OpType::RELU);
	TensorPtr result = a->value()->relu();
	bool needs_grad = GradMode::isEnabled() && a->value()->requires_grad();
	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::RELU, node);
		node->setAdjoint(adjoint);

		adjoint->addDependency(a->adjoint(), a->value()->binarilize());

		graph_->addNode(adjoint);
	}

	return node;
}

NodePtr GraphBuilder::createSumNode(NodePtr a, const std::string& name) {
	auto node = std::make_shared<Node>(name, OpType::SUM);
	TensorPtr result = a->value()->sum();
	bool needs_grad = GradMode::isEnabled() && a->value()->requires_grad();
	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::SUM, node);
		node->setAdjoint(adjoint);

		auto gradient = Tensor::ones(*a->value()->dims);

		// std::cout << "CreateSumNode grad: ";
		// gradient->print(); 
		// std::cout << std::endl;

		adjoint->addDependency(a->adjoint(), gradient);

		graph_->addNode(adjoint);
	}
	return node;
}

NodePtr GraphBuilder::createElementwiseMultiplyNode(NodePtr a, NodePtr b, const std::string& name) {
	// Create forward node
	auto node = std::make_shared<Node>(name, OpType::ELEMENTWISE_MULTIPLY);
	TensorPtr result = a->value()->elementwise_mult(b->value());
	bool needs_grad = GradMode::isEnabled() && (a->value()->requires_grad() || b->value()->requires_grad());
	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {
		// Create adjoint node
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::ELEMENTWISE_MULTIPLY, node);
		node->setAdjoint(adjoint);

		// Add dependencies for adjoints
		if (a->value()->requires_grad()) {
			adjoint->addDependency(a->adjoint(), b->value());
		}

		if (b->value()->requires_grad()) {
			adjoint->addDependency(b->adjoint(), a->value());
		}

		graph_->addNode(adjoint);
	}
	return node;
}

NodePtr GraphBuilder::createTransposeNode(NodePtr a, const std::string& name) {
	auto node = std::make_shared<Node>(name, OpType::TRANSPOSE);
	TensorPtr result = a->value()->transpose();
	bool needs_grad = GradMode::isEnabled() && a->value()->requires_grad();
	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::TRANSPOSE, node);
		node->setAdjoint(adjoint);

		// Transpose operation is its own adjoint
		adjoint->addDependency(a->adjoint(), std::make_shared<Tensor>(1.0f, true));

		graph_->addNode(adjoint);
	}
	return node;
}

NodePtr GraphBuilder::createSumAlongDimensionNode(NodePtr a, int axis, const std::string& name) {
	auto node = std::make_shared<Node>(name, OpType::REDUCED_SUM);
	TensorPtr result = a->value()->reduction_sum(axis);
	bool needs_grad = GradMode::isEnabled() && a->value()->requires_grad();
	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::REDUCED_SUM, node);
		node->setAdjoint(adjoint);

		auto grad = Tensor::ones(*a->value()->dims);

		adjoint->addDependency(a->adjoint(), grad);

		// 		std::cout << "grad: ";
		// grad->print(); 
		//   std::cout << std::endl;

		graph_->addNode(adjoint);
	}
	return node;
}

NodePtr GraphBuilder::createDivideNode(NodePtr a, NodePtr b, const std::string& name) {
	auto node = std::make_shared<Node>(name, OpType::DIVIDE);

	// Forward pass: element-wise division
	TensorPtr result = a->value()->divide(b->value());
	bool needs_grad = GradMode::isEnabled() && (a->value()->requires_grad() || b->value()->requires_grad());
	result->set_requires_grad(needs_grad);
	node->setValue(result);

	if (needs_grad) {
		// Create adjoint node
		auto adjoint = std::make_shared<AdjointNode>(name + "_bar", OpType::DIVIDE, node);
		node->setAdjoint(adjoint);

		// For division z = a/b, the derivatives are:
		// dz/da = 1/b
		// dz/db = -a/b^2

		// Add dependencies for adjoints

		if (a->value()->requires_grad()) {
			// Gradient with respect to a: grad_output * (1/b)
			auto grad_a = std::make_shared<Tensor>(1.0f, true)->divide(b->value());
			adjoint->addDependency(a->adjoint(), grad_a);
		}

		if (b->value()->requires_grad()) {
			// Gradient with respect to b: grad_output * (-a/b^2)
			TensorPtr grad_b = a->value()->elementwise_mult(std::make_shared<Tensor>(-1.0f, true))->divide((b->value()->pow(2)));
			adjoint->addDependency(b->adjoint(), grad_b);
		}

		graph_->addNode(adjoint);
	}

	return node;
}

void GraphBuilder::backward(NodePtr output_node) {
	graph_->backward(output_node->adjoint());
}
