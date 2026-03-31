//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_LONGSHORTTERMMEMORYMODEL_H
#define SEQUENCEPROCESSING_LONGSHORTTERMMEMORYMODEL_H

#include "RecurrentNeuralNetworkModel.h"

class LongShortTermMemoryModel : public RecurrentNeuralNetworkModel {
public:
  explicit LongShortTermMemoryModel(int wordEmbeddingLength);
  void train(vector<Tensor> &trainSet,
             NeuralNetworkParameter &parameters) override;
  ClassificationPerformance test(const vector<Tensor> &testSet) override;
};

#endif // SEQUENCEPROCESSING_LONGSHORTTERMMEMORYMODEL_H
