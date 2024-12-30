#include "Loss.h"
#include "Tensor.cuh"

CrossEntropyLoss::CrossEntropyLoss(GraphBuilder &builder): builder_(builder) {
}

NodePtr CrossEntropyLoss::forward(NodePtr logits, NodePtr targets) {
    auto& config = Config::getInstance();

    // Negative sum across class dimension
    NodePtr loss = builder_.createCrossEntropyNode(logits, targets, 
        "loss"
    );

    loss->setValue(loss->value()->mean());

    auto device_ = config.getDeviceType();
    if(device_ == DeviceType::GPU){
        loss->value()->copyToCPU();
    }

    return loss;
}