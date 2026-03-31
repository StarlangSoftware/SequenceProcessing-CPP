//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "SquareRoot.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>
#include <cmath>

SquareRoot::SquareRoot(double epsilon) { this->epsilon = epsilon; }

Tensor SquareRoot::calculate(const Tensor &tensor) {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      values.push_back(sqrt(epsilon + tensor.getValue({i, j})));
    }
  }
  return Tensor(values, tensor.getShape());
}

Tensor SquareRoot::derivative(const Tensor &tensor, const Tensor &backward) {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      double val = tensor.getValue({i, j});
      values.push_back(1.0 / (2.0 * val));
    }
  }
  return backward.hadamardProduct(Tensor(values, tensor.getShape()));
}

ComputationalNode *
SquareRoot::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                    bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
