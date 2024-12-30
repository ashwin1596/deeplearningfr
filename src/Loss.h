//
// Created by ashwi on 11/12/2024.
//

#ifndef LOSS_H
#define LOSS_H

#include "Graph.h"

class Loss {
public:
	virtual NodePtr forward(NodePtr logits, NodePtr targets) = 0;
	virtual ~Loss() = default;
};

class CrossEntropyLoss : public Loss {
public:
	CrossEntropyLoss(GraphBuilder& builder);

	NodePtr forward(NodePtr logits, NodePtr targets) override;
private:
	GraphBuilder& builder_;
};

#endif //LOSS_H
