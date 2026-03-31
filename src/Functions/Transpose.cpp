//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "Transpose.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>

Tensor Transpose::calculate(const Tensor &tensor) {
  return tensor.transpose({1, 0});
}

Tensor Transpose::derivative(const Tensor &value, const Tensor &backward) {
  return backward.transpose({1, 0});
}

ComputationalNode *
Transpose::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                   bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
