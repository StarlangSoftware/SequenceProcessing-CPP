//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "MultiplyByConstant.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>

MultiplyByConstant::MultiplyByConstant(double constant) {
  this->constant = constant;
}

Tensor MultiplyByConstant::calculate(const Tensor &tensor) {
  vector<double> values;
  vector<double> tensorValues = tensor.getData();
  for (double val : tensorValues) {
    values.push_back(constant * val);
  }
  return Tensor(values, tensor.getShape());
}

Tensor MultiplyByConstant::derivative(const Tensor &tensor,
                                      const Tensor &backward) {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      values.push_back(constant);
    }
  }
  return backward.hadamardProduct(Tensor(values, tensor.getShape()));
}

ComputationalNode *MultiplyByConstant::addToGraph(
    const std::vector<ComputationalNode *> &inputNodes, bool isBiased,
    ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
