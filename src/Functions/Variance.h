//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_VARIANCE_H
#define SEQUENCEPROCESSING_VARIANCE_H

#include <Function/Function.h>

class Variance : public Function {
public:
  Tensor calculate(const Tensor &tensor) override;
  Tensor derivative(const Tensor &tensor, const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // SEQUENCEPROCESSING_VARIANCE_H
