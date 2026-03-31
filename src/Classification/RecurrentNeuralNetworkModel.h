//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_RECURRENTNEURALNETWORKMODEL_H
#define SEQUENCEPROCESSING_RECURRENTNEURALNETWORKMODEL_H

#include <ComputationalGraph.h>
#include <random>
#include <vector>

using namespace std;

#include "../Functions/Switch.h"
#include "../Parameters/RecurrentNeuralNetworkParameter.h"

class RecurrentNeuralNetworkModel : public ComputationalGraph {
protected:
  int wordEmbeddingLength;
  vector<Switch *> switches;
  RecurrentNeuralNetworkParameter *rnnParameters = nullptr;
  vector<int> createInputTensors(const Tensor &instance);
  void trainModel(vector<Tensor> &trainSet, default_random_engine &random);
  int findTimeStep(const vector<Tensor> &trainSet) const;
  vector<int> getClassLabels(ComputationalNode *outputNode) override;

public:
  explicit RecurrentNeuralNetworkModel(int wordEmbeddingLength);
  void train(vector<Tensor> &trainSet,
             NeuralNetworkParameter &parameters) override;
  ClassificationPerformance test(const vector<Tensor> &testSet) override;
};

#endif // SEQUENCEPROCESSING_RECURRENTNEURALNETWORKMODEL_H
