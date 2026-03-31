//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_TRANSFORMER_H
#define SEQUENCEPROCESSING_TRANSFORMER_H

#include "../Parameters/TransformerParameter.h"
#include <ComputationalGraph.h>
#include <Dictionary/VectorizedDictionary.h>

class Transformer : public ComputationalGraph {
private:
  VectorizedDictionary *dictionary;
  TransformerParameter *tParameters = nullptr;
  int startIndex;
  int endIndex;
  Tensor positionalEncoding(const Tensor &tensor,
                            int wordEmbeddingLength) const;
  vector<int> createInputTensors(const Tensor &instance,
                                 ComputationalNode *input1,
                                 ComputationalNode *input2,
                                 int wordEmbeddingLength);
  ComputationalNode *layerNormalization(ComputationalNode *input, bool isInput,
                                        int lnSize[]);
  vector<ComputationalNode *> multiHeadAttention(ComputationalNode *input,
                                                 bool isMasked,
                                                 default_random_engine &random);
  ComputationalNode *feedforwardNeuralNetwork(ComputationalNode *current,
                                              int currentLayerSize,
                                              bool isInput,
                                              default_random_engine &random);
  void setInputNode(int bound, const Vector &vec, ComputationalNode *node);

protected:
  vector<int> getClassLabels(ComputationalNode *outputNode) override;

public:
  Transformer(VectorizedDictionary *dictionary);
  void train(vector<Tensor> &trainSet,
             NeuralNetworkParameter &parameters) override;
  ClassificationPerformance test(const vector<Tensor> &testSet) override;
};

#endif // SEQUENCEPROCESSING_TRANSFORMER_H
