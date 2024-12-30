#ifndef GRAPH_H
#define GRAPH_H
#include <algorithm>
#include <memory>
#include <vector>
#include <unordered_set>

#include "Tensor.cuh"

class Node;
class AdjointNode;
using NodePtr = std::shared_ptr<Node>;
using AdjointNodePtr = std::shared_ptr<AdjointNode>;

enum class OpType {
	VARIABLE,     // For input variables
	MATMUL,       // Matrix multiplication
	ADD,          // Addition
	SUBTRACT,     // Subtraction
	LN,           // Natural logarithm
	SIN,          // Sine function
	RELU,         // Rectified Linear Unit
	SOFTMAX,      // Softmax function
	SUM,          // Sum reduction
	ELEMENTWISE_MULTIPLY,  // Element-wise multiplication
	REDUCED_SUM, // Reduced sum along the given axis
	TRANSPOSE,
	DIVIDE,
	CROSSENTROPY
};

std::string opTypeToString(OpType op_type);

//Forward computation graph node
class Node {
public:
	Node(const std::string &name, OpType op_type, TensorPtr value = std::make_shared<Tensor>());
	const TensorPtr& value() const;
	void setValue(const TensorPtr& value);
	const std::string& name() const;
	void setName(const std::string& name);
	void setAdjoint(const AdjointNodePtr& adjoint);
	AdjointNodePtr adjoint() const;
	OpType op_type() const;
private:
	std::string name_;
	TensorPtr value_;
	AdjointNodePtr adjoint_;
	OpType op_type_;
};

//Adjoint graph node with topological sorting support
class AdjointNode {
public:
	AdjointNode(const std::string& name, OpType op_type, NodePtr primal);
	void addDependency(AdjointNodePtr dep, const TensorPtr& partial_derivative);

	// Process this node's gradient using stored partial derivatives
	void processGradient();

	TensorPtr& get_adjoint();
	void set_adjoint(const TensorPtr& tensor );
	const std::string& name() const;
	void setName(const std::string& name);
	OpType op_type() const;
	void requires_red_sum(bool needs_reduction);
	bool needs_red_sum() const;

	//For topological sorting
	std::vector<std::pair<AdjointNodePtr, TensorPtr>> dependencies_;
	std::unordered_set<AdjointNode*> parents_;
	TensorPtr adjoint_value_;

private:
	std::string name_;
	NodePtr primal_;
	OpType op_type_;
	bool needs_reduction_;
};

// Graph manager that handles topological sorting and backward pass(autodiff)
class ComputationGraph {
public:
	void addNode(AdjointNodePtr node) ;

	// Performs topological sort and computes gradients
	void backward(AdjointNodePtr start_node);

private:
	std::vector<AdjointNodePtr> nodes_;

	//topological sort
	std::vector<AdjointNodePtr> topologicalSort(AdjointNodePtr start_node);
};

class GraphBuilder {
public:
	GraphBuilder();

	NodePtr createVariable(const std::string& name, const TensorPtr& value);

	NodePtr createMatmulNode(NodePtr a, NodePtr b, const std::string& name);
	NodePtr createAddNode(NodePtr a, NodePtr b, const std::string& name);
	NodePtr createLnNode(NodePtr a, const std::string& name);
	NodePtr createSubtractNode(NodePtr a, NodePtr b, const std::string& name);
	NodePtr createReluNode(NodePtr a, const std::string& name);
	NodePtr createSumNode(NodePtr a, const std::string& name);
	NodePtr createElementwiseMultiplyNode(NodePtr a, NodePtr b, const std::string& name);
	NodePtr createTransposeNode(NodePtr a, const std::string& name);
	NodePtr createSumAlongDimensionNode(NodePtr a, int axis, const std::string& name);
	NodePtr createDivideNode(NodePtr a, NodePtr b, const std::string& name);
	NodePtr createCrossEntropyNode(NodePtr a, NodePtr b, const std::string& name);
	void backward(NodePtr output_node);

private:
	std::shared_ptr<ComputationGraph> graph_;
};

#endif //GRAPH_H
