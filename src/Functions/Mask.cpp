//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "Mask.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>
#include <limits>

Tensor Mask::calculate(const Tensor &tensor) {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      if (j > i) {
        values.push_back(-numeric_limits<double>::infinity());
      } else {
        values.push_back(tensor.getValue({i, j}));
      }
    }
  }
  return Tensor(values, tensor.getShape());
}

Tensor Mask::derivative(const Tensor &tensor, const Tensor &backward) {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      values.push_back(1.0);
    }
  }
  return backward.hadamardProduct(Tensor(values, tensor.getShape()));
}

ComputationalNode *
Mask::addToGraph(const std::vector<ComputationalNode *> &inputNodes, bool isBiased,
              ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
