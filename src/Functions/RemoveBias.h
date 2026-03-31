//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_REMOVEBIAS_H
#define SEQUENCEPROCESSING_REMOVEBIAS_H

#include <Function/Function.h>

class RemoveBias : public Function {
public:
  Tensor calculate(const Tensor &matrix) override;
  Tensor derivative(const Tensor &value, const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // SEQUENCEPROCESSING_REMOVEBIAS_H
