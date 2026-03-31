//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "GatedRecurrentUnitModel.h"
#include "../Functions/AdditionByConstant.h"
#include "../Functions/RemoveBias.h"
#include "../Functions/Switch.h"
#include <Function/Negation.h>
#include <Function/Softmax.h>
#include <Function/Tanh.h>
#include <Node/ConcatenatedNode.h>
#include <random>

ClassificationPerformance
GatedRecurrentUnitModel::test(const vector<Tensor> &testSet) {
  return RecurrentNeuralNetworkModel::test(testSet);
}

GatedRecurrentUnitModel::GatedRecurrentUnitModel(int wordEmbeddingLength)
    : RecurrentNeuralNetworkModel(wordEmbeddingLength) {
  this->switches = vector<Switch *>();
}

void GatedRecurrentUnitModel::train(vector<Tensor> &trainSet,
                                    NeuralNetworkParameter &parameters) {
  rnnParameters = dynamic_cast<RecurrentNeuralNetworkParameter *>(&parameters);
  default_random_engine random(rnnParameters->getSeed());
  int timeStep = findTimeStep(trainSet);
  vector<MultiplicationNode *> weights;
  vector<MultiplicationNode *> recurrentWeights;
  int currentLength = wordEmbeddingLength + 1;
  for (int i = 0; i < rnnParameters->size(); i++) {
    for (int j = 0; j < 3; j++) {
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
  vector<ComputationalNode *> outputNodes;
  for (int k = 0; k < timeStep; k++) {
    switches.push_back(new Switch());
    vector<ComputationalNode *> newOldLayers;
    auto *input = new MultiplicationNode(false, true);
    inputNodes.push_back(input);
    ComputationalNode *current = input;
    for (int i = 0; i < rnnParameters->size(); i++) {
      ComputationalNode *aw;
      ComputationalNode *aFunction;
      if (!currentOldLayers.empty()) {
        aw = addMultiplicationEdge(current, weights[i * 3], false);
        ComputationalNode *oWithoutBias =
            addFunctionEdge(currentOldLayers[i], new RemoveBias());
        ComputationalNode *ou =
            addMultiplicationEdge(oWithoutBias, recurrentWeights[i * 3], false);
        ComputationalNode *awOu = addAdditionEdge(aw, ou, false);
        ComputationalNode *zt = addFunctionEdge(
            awOu, rnnParameters->getActivationFunction(i * 2), false);
        aw = addMultiplicationEdge(current, weights[i * 3 + 1], false);
        ou = addMultiplicationEdge(oWithoutBias, recurrentWeights[i * 3 + 1],
                                   false);
        awOu = addAdditionEdge(aw, ou, false);
        ComputationalNode *rt = addFunctionEdge(
            awOu, rnnParameters->getActivationFunction(i * 2 + 1), false);
        aw = addMultiplicationEdge(current, weights[i * 3 + 2], false);
        ComputationalNode *rtHt1 = addNodeEdge(rt, oWithoutBias, false, true);
        ou = addMultiplicationEdge(rtHt1, recurrentWeights[i * 3 + 2], false);
        awOu = addAdditionEdge(aw, ou, false);
        ComputationalNode *hTemp = addFunctionEdge(awOu, new Tanh(), false);
        ComputationalNode *minusZt = addFunctionEdge(zt, new Negation(), false);
        ComputationalNode *oneMinusZt =
            addFunctionEdge(minusZt, new AdditionByConstant(1.0), false);
        aw = addNodeEdge(oneMinusZt, oWithoutBias, false, true);
        ou = addNodeEdge(hTemp, zt, false, true);
        aFunction = addAdditionEdge(aw, ou, true);
      } else {
        aw = addMultiplicationEdge(current, weights[i * 3], false);
        ComputationalNode *zt = addFunctionEdge(
            aw, rnnParameters->getActivationFunction(i * 2), false);
        aw = addMultiplicationEdge(current, weights[i * 3 + 2], false);
        ComputationalNode *hTemp = addFunctionEdge(aw, new Tanh(), false);
        aFunction = addNodeEdge(zt, hTemp, true, true);
      }
      current = aFunction;
      newOldLayers.push_back(aFunction);
    }
    currentOldLayers = newOldLayers;
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
