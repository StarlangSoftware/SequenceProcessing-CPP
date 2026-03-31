//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "RemoveBias.h"
#include <ComputationalGraph.h>
#include <Node/ComputationalNode.h>

Tensor RemoveBias::calculate(const Tensor &matrix) {
  vector<double> data = matrix.getData();
  vector<double> values;
  for (int i = 0; i < (int)data.size() - 1; i++) {
    values.push_back(data[i]);
  }
  return Tensor(values, {1, (int)values.size()});
}

Tensor RemoveBias::derivative(const Tensor &value, const Tensor &backward) {
  vector<double> values = backward.getData();
  vector<double> newValues(values.begin(), values.end());
  newValues.push_back(0.0);
  return Tensor(newValues, {1, (int)newValues.size()});
}

ComputationalNode *
RemoveBias::addToGraph(const std::vector<ComputationalNode *> &inputNodes,
                    bool isBiased, ComputationalGraph *graph) {
  auto *newNode = new ComputationalNode(false, this, isBiased);
  graph->computeIfAbsent(graph->nodeMap, inputNodes[0], newNode);
  graph->computeIfAbsent(graph->reverseNodeMap, newNode, inputNodes[0]);
  return newNode;
}
