//
// Created by Olcay Taner YILDIZ on 9.03.2026.
//

#ifndef SEQUENCEPROCESSING_RECURRENTNEURALNETWORKPARAMETER_H
#define SEQUENCEPROCESSING_RECURRENTNEURALNETWORKPARAMETER_H

#include <vector>
#include <NeuralNetworkParameter.h>
#include <Function/Function.h>

using namespace std;

class RecurrentNeuralNetworkParameter : public NeuralNetworkParameter {
private:
    vector<int> hiddenLayers;
    vector<Function*> functions;
    int classLabelSize;
public:
    RecurrentNeuralNetworkParameter(int seed, int epoch, Optimizer* optimizer, Initialization* initialization,
                                     vector<int> hiddenLayers, vector<Function*> functions, int classLabelSize);
    int size() const;
    int getClassLabelSize() const;
    Function* getActivationFunction(int index) const;
    int getHiddenLayer(int index) const;
};

#endif //SEQUENCEPROCESSING_RECURRENTNEURALNETWORKPARAMETER_H
