//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "LongShortTermMemoryModel.h"
#include "../Functions/RemoveBias.h"
#include "../Functions/Switch.h"
#include <Function/Softmax.h>
#include <Function/Tanh.h>
#include <Node/ConcatenatedNode.h>
#include <random>

ClassificationPerformance
LongShortTermMemoryModel::test(const vector<Tensor> &testSet) {
  return RecurrentNeuralNetworkModel::test(testSet);
}

LongShortTermMemoryModel::LongShortTermMemoryModel(int wordEmbeddingLength)
    : RecurrentNeuralNetworkModel(wordEmbeddingLength) {
  this->switches = vector<Switch *>();
}

void LongShortTermMemoryModel::train(vector<Tensor> &trainSet,
                                     NeuralNetworkParameter &parameters) {
  rnnParameters = dynamic_cast<RecurrentNeuralNetworkParameter *>(&parameters);
  default_random_engine random(rnnParameters->getSeed());
  int timeStep = findTimeStep(trainSet);
  vector<MultiplicationNode *> weights;
  vector<MultiplicationNode *> recurrentWeights;
  int currentLength = wordEmbeddingLength + 1;
  for (int i = 0; i < rnnParameters->size(); i++) {
    for (int j = 0; j < 4; j++) {
      weights.push_back(new MultiplicationNode(
          Tensor(rnnParameters->getInitialization()->initialize(
                     currentLength, rnnParameters->getHiddenLayer(i), random),
                 {currentLength, rnnParameters->getHiddenLayer(i)})));
      recurrentWeights.push_back(new MultiplicationNode(
          Tensor(rnnParameters->getInitialization()->initialize(
                     rnnParameters->getHiddenLayer(i),
                     rnnParameters->getHiddenLayer(i), random),
                 {rnnParameters->getHiddenLayer(i),
                  rnnParameters->getHiddenLayer(i)})));
    }
    currentLength = rnnParameters->getHiddenLayer(i) + 1;
  }
  weights.push_back(new MultiplicationNode(
      Tensor(rnnParameters->getInitialization()->initialize(
                 currentLength, rnnParameters->getClassLabelSize(), random),
             {currentLength, rnnParameters->getClassLabelSize()})));
  vector<ComputationalNode *> currentOldLayers;
  vector<ComputationalNode *> currentOldCValues;
  vector<ComputationalNode *> outputNodes;
  for (int k = 0; k < timeStep; k++) {
    switches.push_back(new Switch());
    vector<ComputationalNode *> newOldLayers;
    vector<ComputationalNode *> newOldCValues;
    auto *input = new MultiplicationNode(false, true);
    inputNodes.push_back(input);
    ComputationalNode *current = input;
    for (int i = 0; i < (int)weights.size() - 1; i += 4) {
      ComputationalNode *aw;
      ComputationalNode *aFunction;
      ComputationalNode *ct;
      if (!currentOldLayers.empty()) {
        aw = addMultiplicationEdge(current, weights[i], false);
        ComputationalNode *oWithoutBias =
            addFunctionEdge(currentOldLayers[i / 4], new RemoveBias());
        ComputationalNode *ou =
            addMultiplicationEdge(oWithoutBias, recurrentWeights[i], false);
        ComputationalNode *awOu = addAdditionEdge(aw, ou, false);
        ComputationalNode *it = addFunctionEdge(
            awOu, rnnParameters->getActivationFunction(i), false);
        aw = addMultiplicationEdge(current, weights[i + 1], false);
        ou =
            addMultiplicationEdge(oWithoutBias, recurrentWeights[i + 1], false);
        awOu = addAdditionEdge(aw, ou, false);
        ComputationalNode *ft = addFunctionEdge(
            awOu, rnnParameters->getActivationFunction(i + 1), false);
        aw = addMultiplicationEdge(current, weights[i + 2], false);
        ou =
            addMultiplicationEdge(oWithoutBias, recurrentWeights[i + 2], false);
        awOu = addAdditionEdge(aw, ou, false);
        ComputationalNode *ot = addFunctionEdge(
            awOu, rnnParameters->getActivationFunction(i + 2), false);
        aw = addMultiplicationEdge(current, weights[i + 3], false);
        ou =
            addMultiplicationEdge(oWithoutBias, recurrentWeights[i + 3], false);
        awOu = addAdditionEdge(aw, ou, false);
        ComputationalNode *cTemp = addFunctionEdge(awOu, new Tanh(), false);
        ComputationalNode *ftCt1 =
            addNodeEdge(ft, currentOldCValues[i / 4], false, true);
        ComputationalNode *itCtTemp = addNodeEdge(it, cTemp, false, true);
        ComputationalNode *cmb = addAdditionEdge(ftCt1, itCtTemp, false);
        ct = addFunctionEdge(cmb, rnnParameters->getActivationFunction(i + 3),
                             false);
        ComputationalNode *tanhCt = addFunctionEdge(ct, new Tanh(), false);
        aFunction = addNodeEdge(tanhCt, ot, true, true);
      } else {
        aw = addMultiplicationEdge(current, weights[i], false);
        ComputationalNode *it =
            addFunctionEdge(aw, rnnParameters->getActivationFunction(i), false);
        aw = addMultiplicationEdge(current, weights[i + 1], false);
        ComputationalNode *ot = addFunctionEdge(
            aw, rnnParameters->getActivationFunction(i + 2), false);
        aw = addMultiplicationEdge(current, weights[i + 3], false);
        ComputationalNode *cTemp = addFunctionEdge(aw, new Tanh(), false);
        ComputationalNode *itCTemp = addNodeEdge(it, cTemp, false, true);
        ct = addFunctionEdge(
            itCTemp, rnnParameters->getActivationFunction(i + 3), false);
        ComputationalNode *tanhCt = addFunctionEdge(ct, new Tanh(), false);
        aFunction = addNodeEdge(tanhCt, ot, true, true);
      }
      current = aFunction;
      newOldLayers.push_back(aFunction);
      newOldCValues.push_back(ct);
    }
    currentOldLayers = newOldLayers;
    currentOldCValues = newOldCValues;
    ComputationalNode *node =
        addMultiplicationEdge(current, weights[weights.size() - 1], false);
    outputNodes.push_back(addFunctionEdge(node, switches[k], false));
  }
  auto *concatenatedNode =
      dynamic_cast<ConcatenatedNode *>(concatEdges(outputNodes, 0));
  outputNode = addFunctionEdge(concatenatedNode, new Softmax(), false);
  auto *classLabelNode = new ComputationalNode(false, false);
  inputNodes.push_back(classLabelNode);
  trainModel(trainSet, random);
}
