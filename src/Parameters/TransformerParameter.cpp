//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "TransformerParameter.h"

TransformerParameter::TransformerParameter(int seed, int epoch, Optimizer* optimizer, Initialization* initialization,
                                           int wordEmbeddingLength, int multiHeadAttentionLength,
                                           int vocabularyLength, double epsilon,
                                           vector<int> inputHiddenLayers, vector<int> outputHiddenLayers,
                                           vector<Function*> inputActivationFunctions,
                                           vector<Function*> outputActivationFunctions,
                                           vector<double> gammaInputValues, vector<double> gammaOutputValues,
                                           vector<double> betaInputValues, vector<double> betaOutputValues)
        : NeuralNetworkParameter(seed, epoch, optimizer, initialization) {
    this->L = wordEmbeddingLength + 1;
    this->N = multiHeadAttentionLength;
    this->V = vocabularyLength;
    this->epsilon = epsilon;
    this->inputHiddenLayers = inputHiddenLayers;
    this->outputHiddenLayers = outputHiddenLayers;
    this->inputFunctions = inputActivationFunctions;
    this->outputFunctions = outputActivationFunctions;
    this->gammaInputValues = gammaInputValues;
    this->gammaOutputValues = gammaOutputValues;
    this->betaInputValues = betaInputValues;
    this->betaOutputValues = betaOutputValues;
}

double TransformerParameter::getGammaInputValue(int index) const {
    return gammaInputValues[index];
}

double TransformerParameter::getGammaOutputValue(int index) const {
    return gammaOutputValues[index];
}

double TransformerParameter::getBetaInputValue(int index) const {
    return betaInputValues[index];
}

double TransformerParameter::getBetaOutputValue(int index) const {
    return betaOutputValues[index];
}

double TransformerParameter::getEpsilon() const {
    return epsilon;
}

int TransformerParameter::getDk() const {
    return L / N;
}

int TransformerParameter::getL() const {
    return L;
}

int TransformerParameter::getN() const {
    return N;
}

int TransformerParameter::getV() const {
    return V;
}

int TransformerParameter::getInputHiddenLayer(int index) const {
    return inputHiddenLayers[index];
}

int TransformerParameter::getOutputHiddenLayer(int index) const {
    return outputHiddenLayers[index];
}

Function* TransformerParameter::getInputActivationFunction(int index) const {
    return inputFunctions[index];
}

Function* TransformerParameter::getOutputActivationFunction(int index) const {
    return outputFunctions[index];
}

int TransformerParameter::getInputSize() const {
    return inputHiddenLayers.size();
}

int TransformerParameter::getOutputSize() const {
    return outputHiddenLayers.size();
}
