//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_GATEDRECURRENTUNITMODEL_H
#define SEQUENCEPROCESSING_GATEDRECURRENTUNITMODEL_H

#include "RecurrentNeuralNetworkModel.h"

class GatedRecurrentUnitModel : public RecurrentNeuralNetworkModel {
public:
  explicit GatedRecurrentUnitModel(int wordEmbeddingLength);
  void train(vector<Tensor> &trainSet,
             NeuralNetworkParameter &parameters) override;
  ClassificationPerformance test(const vector<Tensor> &testSet) override;
};

#endif // SEQUENCEPROCESSING_GATEDRECURRENTUNITMODEL_H
