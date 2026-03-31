//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_TRANSFORMERPARAMETER_H
#define SEQUENCEPROCESSING_TRANSFORMERPARAMETER_H

#include <vector>
#include <NeuralNetworkParameter.h>
#include <Function/Function.h>

using namespace std;

class TransformerParameter : public NeuralNetworkParameter {
private:
    int L;
    int N;
    int V;
    double epsilon;
    vector<int> inputHiddenLayers;
    vector<int> outputHiddenLayers;
    vector<Function*> inputFunctions;
    vector<Function*> outputFunctions;
    vector<double> gammaInputValues;
    vector<double> gammaOutputValues;
    vector<double> betaInputValues;
    vector<double> betaOutputValues;
public:
    TransformerParameter(int seed, int epoch, Optimizer* optimizer, Initialization* initialization,
                         int wordEmbeddingLength, int multiHeadAttentionLength, int vocabularyLength,
                         double epsilon,
                         vector<int> inputHiddenLayers, vector<int> outputHiddenLayers,
                         vector<Function*> inputActivationFunctions, vector<Function*> outputActivationFunctions,
                         vector<double> gammaInputValues, vector<double> gammaOutputValues,
                         vector<double> betaInputValues, vector<double> betaOutputValues);
    double getGammaInputValue(int index) const;
    double getGammaOutputValue(int index) const;
    double getBetaInputValue(int index) const;
    double getBetaOutputValue(int index) const;
    double getEpsilon() const;
    int getDk() const;
    int getL() const;
    int getN() const;
    int getV() const;
    int getInputHiddenLayer(int index) const;
    int getOutputHiddenLayer(int index) const;
    Function* getInputActivationFunction(int index) const;
    Function* getOutputActivationFunction(int index) const;
    int getInputSize() const;
    int getOutputSize() const;
};

#endif //SEQUENCEPROCESSING_TRANSFORMERPARAMETER_H
