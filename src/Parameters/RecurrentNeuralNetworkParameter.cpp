//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#include "RecurrentNeuralNetworkParameter.h"

RecurrentNeuralNetworkParameter::RecurrentNeuralNetworkParameter(int seed, int epoch, Optimizer* optimizer,
                                                                   Initialization* initialization,
                                                                   vector<int> hiddenLayers,
                                                                   vector<Function*> functions,
                                                                   int classLabelSize)
        : NeuralNetworkParameter(seed, epoch, optimizer, initialization) {
    this->hiddenLayers = hiddenLayers;
    this->functions = functions;
    this->classLabelSize = classLabelSize;
}

int RecurrentNeuralNetworkParameter::size() const {
    return hiddenLayers.size();
}

int RecurrentNeuralNetworkParameter::getClassLabelSize() const {
    return classLabelSize;
}

Function* RecurrentNeuralNetworkParameter::getActivationFunction(int index) const {
    return functions[index];
}

int RecurrentNeuralNetworkParameter::getHiddenLayer(int index) const {
    return hiddenLayers[index];
}
