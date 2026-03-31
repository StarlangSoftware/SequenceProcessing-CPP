//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "AdditionByConstant.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>

AdditionByConstant::AdditionByConstant(double constant) {
  this->constant = constant;
}

Tensor AdditionByConstant::calculate(const Tensor &tensor) {
  vector<double> values;
  vector<double> tensorValues = tensor.getData();
  for (double val : tensorValues) {
    values.push_back(val + constant);
  }
  return Tensor(values, tensor.getShape());
}

Tensor AdditionByConstant::derivative(const Tensor &tensor,
                                      const Tensor &backward) {
  return backward;
}

ComputationalNode *
AdditionByConstant::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                            bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
