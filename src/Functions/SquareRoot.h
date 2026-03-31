//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_SQUAREROOT_H
#define SEQUENCEPROCESSING_SQUAREROOT_H

#include <Function/Function.h>

class SquareRoot : public Function {
private:
  double epsilon;

public:
  explicit SquareRoot(double epsilon);
  Tensor calculate(const Tensor &tensor) override;
  Tensor derivative(const Tensor &tensor, const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // SEQUENCEPROCESSING_SQUAREROOT_H
