//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "Transformer.h"
#include "../Functions/Inverse.h"
#include "../Functions/Mask.h"
#include "../Functions/Mean.h"
#include "../Functions/MultiplyByConstant.h"
#include "../Functions/SquareRoot.h"
#include "../Functions/Transpose.h"
#include "../Functions/Variance.h"
#include <Dictionary/VectorizedWord.h>
#include <Function/Negation.h>
#include <Function/Softmax.h>
#include <Node/ConcatenatedNode.h>
#include <cmath>
#include <limits>
#include <random>

Transformer::Transformer(VectorizedDictionary *dictionary) {
  this->dictionary = dictionary;
  for (int k = 0; k < dictionary->size(); k++) {
    if (dictionary->getWord(k)->getName() == "<S>") {
      startIndex = k;
    } else if (dictionary->getWord(k)->getName() == "</S>") {
      endIndex = k;
    }
  }
}

Tensor Transformer::positionalEncoding(const Tensor &tensor,
                                       int wordEmbeddingLength) const {
  vector<double> values;
  for (int i = 0; i < tensor.getShape()[0]; i++) {
    for (int j = 0; j < tensor.getShape()[1]; j++) {
      double val = tensor.getValue({i, j});
      if (j % 2 == 0) {
        values.push_back(
            val + sin((i + 1.0) / pow(10000, (j + 0.0) / wordEmbeddingLength)));
      } else {
        values.push_back(
            val + cos((i + 1.0) / pow(10000, (j - 1.0) / wordEmbeddingLength)));
      }
    }
  }
  return Tensor(values, tensor.getShape());
}

vector<int> Transformer::createInputTensors(const Tensor &instance,
                                            ComputationalNode *input1,
                                            ComputationalNode *input2,
                                            int wordEmbeddingLength) {
  bool isOutput = false;
  int curLength = 0;
  vector<int> classLabels;
  vector<double> values;
  for (int i = 0; i < instance.getShape()[0]; i++) {
    double val = instance.getValue({i});
    if (val == numeric_limits<double>::max()) {
      isOutput = true;
      input1->setValue(Tensor(
          values, {curLength / wordEmbeddingLength, wordEmbeddingLength}));
      input1->setValue(
          positionalEncoding(input1->getValue(), wordEmbeddingLength));
      curLength = 0;
      values.clear();
    } else if (isOutput) {
      if ((curLength + 1) % (wordEmbeddingLength + 1) == 0) {
        classLabels.push_back((int)val);
      } else {
        values.push_back(val);
      }
      curLength++;
    } else {
      values.push_back(val);
      curLength++;
    }
  }
  input2->setValue(Tensor(
      values, {(int)values.size() / wordEmbeddingLength, wordEmbeddingLength}));
  input2->setValue(positionalEncoding(input2->getValue(), wordEmbeddingLength));
  return classLabels;
}

ComputationalNode *Transformer::layerNormalization(ComputationalNode *input,
                                                   bool isInput, int lnSize[]) {
  vector<double> data;
  ComputationalNode *inputC1Mean = addFunctionEdge(input, new Mean(), false);
  ComputationalNode *mean1Minus =
      addFunctionEdge(inputC1Mean, new Negation(), false);
  ComputationalNode *inputC1Mean1Minus =
      addAdditionEdge(input, mean1Minus, false);
  ComputationalNode *variance1 =
      addFunctionEdge(inputC1Mean1Minus, new Variance(), false);
  ComputationalNode *rootVariance1 = addFunctionEdge(
      variance1, new SquareRoot(tParameters->getEpsilon()), false);
  ComputationalNode *inverseRootVariance1 =
      addFunctionEdge(rootVariance1, new Inverse(), false);
  ComputationalNode *lnValue1 =
      addNodeEdge(inputC1Mean1Minus, inverseRootVariance1, false, true);
  if (isInput) {
    for (int j = 0; j < tParameters->getL(); j++) {
      data.push_back(tParameters->getGammaInputValue(lnSize[0]));
    }
    lnSize[0]++;
  } else {
    for (int j = 0; j < tParameters->getL(); j++) {
      data.push_back(tParameters->getGammaOutputValue(lnSize[1]));
    }
    lnSize[1]++;
  }
  auto *gammaInput1 = new MultiplicationNode(
      true, false, Tensor(data, {1, tParameters->getL()}), true);
  ComputationalNode *lnValue1GammaInput1 =
      addNodeEdge(lnValue1, (ComputationalNode *)gammaInput1, false, true);
  data.clear();
  if (isInput) {
    for (int j = 0; j < tParameters->getL(); j++) {
      data.push_back(tParameters->getBetaInputValue(lnSize[2]));
    }
    lnSize[2]++;
  } else {
    for (int j = 0; j < tParameters->getL(); j++) {
      data.push_back(tParameters->getBetaOutputValue(lnSize[3]));
    }
    lnSize[3]++;
  }
  auto *betaInput1 = new ComputationalNode(
      true, false, nullptr, Tensor(data, {1, tParameters->getL()}));
  return addAdditionEdge(lnValue1GammaInput1, betaInput1, false);
}

vector<ComputationalNode *>
Transformer::multiHeadAttention(ComputationalNode *input, bool isMasked,
                                default_random_engine &random) {
  vector<ComputationalNode *> nodes;
  for (int i = 0; i < tParameters->getN(); i++) {
    auto *wk = new MultiplicationNode(
        Tensor(tParameters->getInitialization()->initialize(
                   tParameters->getL(), tParameters->getDk(), random),
               {tParameters->getL(), tParameters->getDk()}));
    ComputationalNode *k =
        addNodeEdge(input, (ComputationalNode *)wk, false, false);
    auto *wq = new MultiplicationNode(
        Tensor(tParameters->getInitialization()->initialize(
                   tParameters->getL(), tParameters->getDk(), random),
               {tParameters->getL(), tParameters->getDk()}));
    ComputationalNode *q =
        addNodeEdge(input, (ComputationalNode *)wq, false, false);
    auto *wv = new MultiplicationNode(
        Tensor(tParameters->getInitialization()->initialize(
                   tParameters->getL(), tParameters->getDk(), random),
               {tParameters->getL(), tParameters->getDk()}));
    ComputationalNode *v =
        addNodeEdge(input, (ComputationalNode *)wv, false, false);
    ComputationalNode *kTranspose = addFunctionEdge(k, new Transpose(), false);
    ComputationalNode *qk = addNodeEdge(q, kTranspose, false, false);
    ComputationalNode *qkDk = addFunctionEdge(
        qk, new MultiplyByConstant(1.0 / sqrt(tParameters->getDk())), false);
    ComputationalNode *sQkDk;
    if (isMasked) {
      ComputationalNode *mQkDk = addFunctionEdge(qkDk, new Mask(), false);
      sQkDk = addFunctionEdge(mQkDk, new Softmax(), false);
    } else {
      sQkDk = addFunctionEdge(qkDk, new Softmax(), false);
    }
    ComputationalNode *attention = addNodeEdge(sQkDk, v, false, false);
    nodes.push_back(attention);
  }
  return nodes;
}

ComputationalNode *
Transformer::feedforwardNeuralNetwork(ComputationalNode *current,
                                      int currentLayerSize, bool isInput,
                                      default_random_engine &random) {
  int size;
  if (isInput) {
    size = tParameters->getInputSize();
  } else {
    size = tParameters->getOutputSize();
  }
  for (int i = 0; i < size; i++) {
    if (isInput) {
      auto *hiddenWeight = new MultiplicationNode(Tensor(
          tParameters->getInitialization()->initialize(
              currentLayerSize, tParameters->getInputHiddenLayer(i), random),
          {currentLayerSize, tParameters->getInputHiddenLayer(i)}));
      ComputationalNode *hiddenLayer =
          addMultiplicationEdge(current, hiddenWeight, false);
      current = addFunctionEdge(
          hiddenLayer, tParameters->getInputActivationFunction(i), true);
      currentLayerSize = tParameters->getInputHiddenLayer(i) + 1;
    } else {
      auto *hiddenWeight = new MultiplicationNode(Tensor(
          tParameters->getInitialization()->initialize(
              currentLayerSize, tParameters->getOutputHiddenLayer(i), random),
          {currentLayerSize, tParameters->getOutputHiddenLayer(i)}));
      ComputationalNode *hiddenLayer =
          addMultiplicationEdge(current, hiddenWeight, false);
      current = addFunctionEdge(
          hiddenLayer, tParameters->getOutputActivationFunction(i), true);
      currentLayerSize = tParameters->getOutputHiddenLayer(i) + 1;
    }
  }
  auto *outputWeight = new MultiplicationNode(
      Tensor(tParameters->getInitialization()->initialize(
                 currentLayerSize, tParameters->getL(), random),
             {currentLayerSize, tParameters->getL()}));
  ComputationalNode *outputLayer =
      addNodeEdge(current, (ComputationalNode *)outputWeight, false, false);
  return addFunctionEdge(outputLayer, new Softmax(), false);
}

void Transformer::train(vector<Tensor> &trainSet,
                        NeuralNetworkParameter &parameters) {
  tParameters = dynamic_cast<TransformerParameter *>(&parameters);
  int lnSize[4] = {0, 0, 0, 0};
  default_random_engine random(tParameters->getSeed());
  // Encoder Block
  auto *input1 = new MultiplicationNode(false, true);
  inputNodes.push_back(input1);
  auto *concatenatedNode1 = dynamic_cast<ConcatenatedNode *>(
      concatEdges(multiHeadAttention(input1, false, random), 1));
  auto *we = new MultiplicationNode(
      Tensor(tParameters->getInitialization()->initialize(
                 tParameters->getL(), tParameters->getL(), random),
             {tParameters->getL(), tParameters->getL()}));
  ComputationalNode *c1 = addNodeEdge((ComputationalNode *)concatenatedNode1,
                                      (ComputationalNode *)we, false, false);
  ComputationalNode *inputC1 = addAdditionEdge(input1, c1, false);
  ComputationalNode *y1 = layerNormalization(inputC1, true, lnSize);
  ComputationalNode *oe = addAdditionEdge(
      feedforwardNeuralNetwork(y1, tParameters->getL(), true, random), y1,
      false);
  ComputationalNode *encoder = layerNormalization(oe, true, lnSize);
  // Decoder Block
  auto *input2 = new MultiplicationNode(false, true);
  inputNodes.push_back(input2);
  auto *concatenatedNode2 = dynamic_cast<ConcatenatedNode *>(
      concatEdges(multiHeadAttention(input2, true, random), 1));
  auto *wd1 = new MultiplicationNode(
      Tensor(tParameters->getInitialization()->initialize(
                 tParameters->getL(), tParameters->getL(), random),
             {tParameters->getL(), tParameters->getL()}));
  ComputationalNode *c2 = addNodeEdge((ComputationalNode *)concatenatedNode2,
                                      (ComputationalNode *)wd1, false, false);
  ComputationalNode *inputC2 = addAdditionEdge(input2, c2, false);
  ComputationalNode *cd2 = layerNormalization(inputC2, false, lnSize);
  vector<ComputationalNode *> nodes;
  for (int i = 0; i < tParameters->getN(); i++) {
    auto *wk = new MultiplicationNode(
        Tensor(tParameters->getInitialization()->initialize(
                   tParameters->getL(), tParameters->getDk(), random),
               {tParameters->getL(), tParameters->getDk()}));
    ComputationalNode *k =
        addNodeEdge(encoder, (ComputationalNode *)wk, false, false);
    auto *wq = new MultiplicationNode(
        Tensor(tParameters->getInitialization()->initialize(
                   tParameters->getL(), tParameters->getDk(), random),
               {tParameters->getL(), tParameters->getDk()}));
    ComputationalNode *q =
        addNodeEdge(cd2, (ComputationalNode *)wq, false, false);
    auto *wv = new MultiplicationNode(
        Tensor(tParameters->getInitialization()->initialize(
                   tParameters->getL(), tParameters->getDk(), random),
               {tParameters->getL(), tParameters->getDk()}));
    ComputationalNode *v =
        addNodeEdge(encoder, (ComputationalNode *)wv, false, false);
    ComputationalNode *kTranspose = addFunctionEdge(k, new Transpose(), false);
    ComputationalNode *qk = addNodeEdge(q, kTranspose, false, false);
    ComputationalNode *qkDk = addFunctionEdge(
        qk, new MultiplyByConstant(1.0 / sqrt(tParameters->getDk())), false);
    ComputationalNode *sQkDk = addFunctionEdge(qkDk, new Softmax(), false);
    ComputationalNode *attention = addNodeEdge(sQkDk, v, false, false);
    nodes.push_back(attention);
  }
  auto *concatenatedNode3 =
      dynamic_cast<ConcatenatedNode *>(concatEdges(nodes, 1));
  auto *wd2 = new MultiplicationNode(
      Tensor(tParameters->getInitialization()->initialize(
                 tParameters->getL(), tParameters->getL(), random),
             {tParameters->getL(), tParameters->getL()}));
  ComputationalNode *cd3 = addNodeEdge((ComputationalNode *)concatenatedNode3,
                                       (ComputationalNode *)wd2, false, false);
  ComputationalNode *cd3cd2 = addAdditionEdge(cd2, cd3, false);
  ComputationalNode *yd1 = layerNormalization(cd3cd2, false, lnSize);
  ComputationalNode *od =
      feedforwardNeuralNetwork(yd1, tParameters->getL(), false, random);
  ComputationalNode *oy = addAdditionEdge(od, yd1, false);
  ComputationalNode *d = layerNormalization(oy, false, lnSize);
  auto *wdo = new MultiplicationNode(
      Tensor(tParameters->getInitialization()->initialize(
                 tParameters->getL(), tParameters->getV(), random),
             {tParameters->getL(), tParameters->getV()}));
  ComputationalNode *decoder =
      addNodeEdge(d, (ComputationalNode *)wdo, false, false);
  outputNode = addFunctionEdge(decoder, new Softmax(), false);
  auto *classLabelNode = new ComputationalNode(false, false);
  inputNodes.push_back(classLabelNode);
  // Training
  for (int i = 0; i < tParameters->getEpoch(); i++) {
    for (int j = 0; j < (int)trainSet.size(); j++) {
      uniform_int_distribution<int> dist(0, (int)trainSet.size() - 1);
      int i1 = dist(random);
      int i2 = dist(random);
      Tensor tmp = trainSet[i1];
      trainSet[i1] = trainSet[i2];
      trainSet[i2] = tmp;
    }
    for (const Tensor &instance : trainSet) {
      vector<int> classLabels = createInputTensors(
          instance, inputNodes[0], inputNodes[1], tParameters->getL() - 1);
      vector<double> classLabelValues;
      for (int classLabel : classLabels) {
        for (int j = 0; j < tParameters->getV(); j++) {
          if (j == classLabel) {
            classLabelValues.push_back(1.0);
          } else {
            classLabelValues.push_back(0.0);
          }
        }
      }
      inputNodes[2]->setValue(Tensor(
          classLabelValues, {(int)classLabels.size(), tParameters->getV()}));
      vector<int> classLabelIndex = forwardCalculation();
      backpropagation(tParameters->getOptimizer(), classLabelIndex);
    }
    tParameters->getOptimizer()->setLearningRate();
  }
}

void Transformer::setInputNode(int bound, const Vector &vec,
                               ComputationalNode *node) {
  vector<double> data;
  if (!node->isValueNull()) {
    data = node->getValue().getData();
  }
  for (int i = 0; i < vec.getSize(); i++) {
    if (i % 2 == 0) {
      data.push_back(
          vec.getValue(i) +
          sin((bound + 0.0) / pow(10000, (i + 0.0) / vec.getSize())));
    } else {
      data.push_back(
          vec.getValue(i) +
          cos((bound + 0.0) / pow(10000, (i - 1.0) / vec.getSize())));
    }
  }
  node->setValue(Tensor(data, {bound + 1, (int)vec.getSize()}));
}

vector<int> Transformer::getClassLabels(ComputationalNode *outputNode) {
  vector<int> classLabels;
  Tensor value = outputNode->getValue();
  for (int i = 0; i < value.getShape()[0]; i++) {
    double max = -numeric_limits<double>::max();
    int index = -1;
    for (int j = 0; j < value.getShape()[1]; j++) {
      if (value.getValue({i, j}) > max) {
        max = value.getValue({i, j});
        index = j;
      }
    }
    classLabels.push_back(index);
  }
  return classLabels;
}

ClassificationPerformance Transformer::test(const vector<Tensor> &testSet) {
  int count = 0, total = 0;
  for (const Tensor &instance : testSet) {
    vector<int> classLabels;
    vector<int> goldClassLabels = createInputTensors(
        instance, inputNodes[0], new ComputationalNode(false, false),
        (int)((VectorizedWord *)dictionary->getWord((int)0))
            ->getVector()
            .getSize());
    int j = 1;
    int currentWordIndex = startIndex;
    do {
      setInputNode(j,
                   ((VectorizedWord *)dictionary->getWord(currentWordIndex))
                       ->getVector(),
                   inputNodes[1]);
      classLabels = predict();
      if ((int)goldClassLabels.size() >= (int)classLabels.size() &&
          classLabels[classLabels.size() - 1] ==
              goldClassLabels[classLabels.size() - 1]) {
        count++;
      }
      total++;
      j++;
      currentWordIndex = classLabels[classLabels.size() - 1];
    } while (currentWordIndex != endIndex);
    if ((int)classLabels.size() < (int)goldClassLabels.size()) {
      total += (int)goldClassLabels.size() - (int)classLabels.size();
    }
  }
  return ClassificationPerformance((count + 0.00) / total);
}
