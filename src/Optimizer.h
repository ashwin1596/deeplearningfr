#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Graph.h"

class optimizer {
public:
	virtual void step() = 0;
	virtual void zero_grad() = 0;
	virtual ~optimizer() = default;
};

// SGD optimizer implementation
class SGD : public optimizer {
public:
	SGD(std::vector<NodePtr> parameters, float learning_rate);
	void step() override;

	void zero_grad() override;

private:
	std::vector<NodePtr> parameters_;
	float learning_rate_;
};

#endif //OPTIMIZER_H
