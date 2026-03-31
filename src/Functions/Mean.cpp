//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "Mean.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>

Tensor Mean::calculate(const Tensor &tensor) {
  vector<double> values;
  vector<double> means;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    double total = 0.0;
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      total += tensor.getValue({i, j});
    }
    means.push_back(total / tensor.getShape()[1]);
  }
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      values.push_back(means[i]);
    }
  }
  return Tensor(values, tensor.getShape());
}

Tensor Mean::derivative(const Tensor &tensor, const Tensor &backward) {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      values.push_back(1.0 / tensor.getShape()[1]);
    }
  }
  return backward.hadamardProduct(Tensor(values, tensor.getShape()));
}

ComputationalNode *
Mean::addToGraph(const std::vector<ComputationalNode *> &inputNodes, bool isBiased,
              ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
