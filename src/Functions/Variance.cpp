//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "Variance.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>
#include <cmath>

Tensor Variance::calculate(const Tensor &tensor) {
  vector<double> values;
  vector<double> variances;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    double total = 0.0;
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      total += pow(tensor.getValue({i, j}), 2);
    }
    variances.push_back(total / tensor.getShape()[1]);
  }
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      values.push_back(variances[i]);
    }
  }
  return Tensor(values, tensor.getShape());
}

Tensor Variance::derivative(const Tensor &tensor, const Tensor &backward) {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      values.push_back(2.0 *
                       sqrt(tensor.getShape()[1] * tensor.getValue({i, j})) /
                       tensor.getShape()[1]);
    }
  }
  return backward.hadamardProduct(Tensor(values, tensor.getShape()));
}

ComputationalNode *
Variance::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                  bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
