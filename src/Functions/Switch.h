//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_SWITCH_H
#define SEQUENCEPROCESSING_SWITCH_H

#include <Function/Function.h>

class Switch : public Function {
private:
  bool turn;

public:
  Switch();
  void setTurn(bool turn);
  Tensor calculate(const Tensor &matrix) override;
  Tensor derivative(const Tensor &value, const Tensor &backward) override;
  ComputationalNode *addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                             bool isBiased, ComputationalGraph *graph) override;
};

#endif // SEQUENCEPROCESSING_SWITCH_H
