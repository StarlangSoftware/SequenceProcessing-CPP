//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "RecurrentNeuralNetworkModel.h"
#include "../Functions/RemoveBias.h"
#include <Function/Softmax.h>
#include <Node/ConcatenatedNode.h>
#include <limits>
#include <random>

RecurrentNeuralNetworkModel::RecurrentNeuralNetworkModel(
    int wordEmbeddingLength) {
  this->wordEmbeddingLength = wordEmbeddingLength;
}

vector<int>
RecurrentNeuralNetworkModel::createInputTensors(const Tensor &instance) {
  vector<int> classLabels;
  int timeStep = instance.getShape()[0] / (wordEmbeddingLength + 1);
  int j = 0;
  for (int i = 0; i < (int)inputNodes.size() - 1; i++) {
    if (i < timeStep) {
      switches[i]->setTurn(true);
      vector<double> values;
      for (int k = 0; k < wordEmbeddingLength; k++) {
        values.push_back(instance.getValue({j}));
        j++;
      }
      classLabels.push_back((int)instance.getValue({j}));
      j++;
      inputNodes[i]->setValue(Tensor(values, {1, (int)values.size()}));
    } else {
      switches[i]->setTurn(false);
      vector<double> values;
      for (int k = 0; k < wordEmbeddingLength; k++) {
        values.push_back(0.0);
        j++;
      }
      classLabels.push_back(0);
      j++;
      inputNodes[i]->setValue(Tensor(values, {1, (int)values.size()}));
    }
  }
  return classLabels;
}

void RecurrentNeuralNetworkModel::trainModel(vector<Tensor> &trainSet,
                                             default_random_engine &random) {
  for (int i = 0; i < rnnParameters->getEpoch(); i++) {
    for (int j = 0; j < (int)trainSet.size(); j++) {
      uniform_int_distribution<int> dist(0, (int)trainSet.size() - 1);
      int i1 = dist(random);
      int i2 = dist(random);
      Tensor tmp = trainSet[i1];
      trainSet[i1] = trainSet[i2];
      trainSet[i2] = tmp;
    }
    for (const Tensor &instance : trainSet) {
      vector<int> classLabels = createInputTensors(instance);
      vector<double> classLabelValues;
      for (int classLabel : classLabels) {
        for (int in = 0; in < rnnParameters->getClassLabelSize(); in++) {
          if (in == classLabel) {
            classLabelValues.push_back(1.0);
          } else {
            classLabelValues.push_back(0.0);
          }
        }
      }
      inputNodes[inputNodes.size() - 1]->setValue(
          Tensor(classLabelValues, {(int)classLabels.size(),
                                    rnnParameters->getClassLabelSize()}));
      vector<int> classLabelIndex = forwardCalculation();
      backpropagation(rnnParameters->getOptimizer(), classLabelIndex);
    }
    rnnParameters->getOptimizer()->setLearningRate();
  }
}

int RecurrentNeuralNetworkModel::findTimeStep(
    const vector<Tensor> &trainSet) const {
  int timeStep = -1;
  for (const Tensor &tensor : trainSet) {
    int size = tensor.getShape()[0];
    if (timeStep < size / (wordEmbeddingLength + 1)) {
      timeStep = size / (wordEmbeddingLength + 1);
    }
  }
  return timeStep;
}

void RecurrentNeuralNetworkModel::train(vector<Tensor> &trainSet,
                                        NeuralNetworkParameter &parameters) {
  rnnParameters = dynamic_cast<RecurrentNeuralNetworkParameter *>(&parameters);
  default_random_engine random(rnnParameters->getSeed());
  int timeStep = findTimeStep(trainSet);
  vector<MultiplicationNode *> weights;
  vector<MultiplicationNode *> recurrentWeights;
  int currentLength = wordEmbeddingLength + 1;
  for (int i = 0; i < rnnParameters->size(); i++) {
    weights.push_back(new MultiplicationNode(
        Tensor(rnnParameters->getInitialization()->initialize(
                   currentLength, rnnParameters->getHiddenLayer(i), random),
               {currentLength, rnnParameters->getHiddenLayer(i)})));
    recurrentWeights.push_back(new MultiplicationNode(Tensor(
        rnnParameters->getInitialization()->initialize(
            rnnParameters->getHiddenLayer(i), rnnParameters->getHiddenLayer(i),
            random),
        {rnnParameters->getHiddenLayer(i), rnnParameters->getHiddenLayer(i)})));
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
        aw = addMultiplicationEdge(current, weights[i], false);
        ComputationalNode *oWithoutBias =
            addFunctionEdge(currentOldLayers[i], new RemoveBias());
        ComputationalNode *ou =
            addMultiplicationEdge(oWithoutBias, recurrentWeights[i], false);
        ComputationalNode *a = addAdditionEdge(aw, ou, false);
        aFunction =
            addFunctionEdge(a, rnnParameters->getActivationFunction(i), true);
      } else {
        aw = addMultiplicationEdge(current, weights[i], false);
        aFunction =
            addFunctionEdge(aw, rnnParameters->getActivationFunction(i), true);
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

vector<int>
RecurrentNeuralNetworkModel::getClassLabels(ComputationalNode *outputNode) {
  vector<int> classLabels;
  for (int i = 0; i < outputNode->getValue().getShape()[0]; i++) {
    int index = -1;
    double max = -numeric_limits<double>::max();
    for (int j = 0; j < outputNode->getValue().getShape()[1]; j++) {
      if (max < outputNode->getValue().getValue({i, j})) {
        max = outputNode->getValue().getValue({i, j});
        index = j;
      }
    }
    classLabels.push_back(index);
  }
  return classLabels;
}

ClassificationPerformance
RecurrentNeuralNetworkModel::test(const vector<Tensor> &testSet) {
  int count = 0, total = 0;
  for (const Tensor &instance : testSet) {
    vector<int> goldClassLabels = createInputTensors(instance);
    vector<int> classLabels = predict();
    for (int j = 0;
         j < (int)(instance.getShape()[0] / (wordEmbeddingLength + 1)); j++) {
      if (goldClassLabels[j] == classLabels[j]) {
        count++;
      }
      total++;
    }
  }
  return ClassificationPerformance((count + 0.0) / total);
}
