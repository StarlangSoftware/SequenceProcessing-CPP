//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "Switch.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>

Switch::Switch() { this->turn = true; }

void Switch::setTurn(bool turn) { this->turn = turn; }

Tensor Switch::calculate(const Tensor &matrix) {
  if (turn) {
    return matrix;
  }
  vector<double> values;
  int size = 1;
  for (int i = 0; i < (int)matrix.getShape().size(); i++) {
    size *= matrix.getShape()[i];
  }
  for (int i = 0; i < size; i++) {
    values.push_back(0.0);
  }
  return Tensor(values, matrix.getShape());
}

Tensor Switch::derivative(const Tensor &value, const Tensor &backward) {
  if (turn) {
    return backward;
  }
  return calculate(value);
}

ComputationalNode *
Switch::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
