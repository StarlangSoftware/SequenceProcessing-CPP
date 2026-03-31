//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "Inverse.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>
#include <cmath>

Tensor Inverse::calculate(const Tensor &tensor) {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      values.push_back(1.0 / tensor.getValue({i, j}));
    }
  }
  return Tensor(values, tensor.getShape());
}

Tensor Inverse::derivative(const Tensor &tensor, const Tensor &backward) {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      values.push_back(-pow(tensor.getValue({i, j}), 2));
    }
  }
  return backward.hadamardProduct(Tensor(values, tensor.getShape()));
}

ComputationalNode *
Inverse::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                 bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
