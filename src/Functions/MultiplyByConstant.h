//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_MULTIPLYBYCONSTANT_H
#define SEQUENCEPROCESSING_MULTIPLYBYCONSTANT_H

#include <Function/Function.h>

class MultiplyByConstant : public Function {
private:
  double constant;

public:
  explicit MultiplyByConstant(double constant);
  Tensor calculate(const Tensor &tensor) override;
  Tensor derivative(const Tensor &tensor, const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // SEQUENCEPROCESSING_MULTIPLYBYCONSTANT_H
