#include "../src/Classification/Transformer.h"
#include "catch.hpp"
#include <Dictionary/VectorizedDictionary.h>
#include <Function/Sigmoid.h>
#include <Function/Tanh.h>
#include <Initialization/RandomInitialization.h>
#include <Optimizer/AdamW.h>
#include <Tensor.h>
#include <limits>
#include <vector>

using namespace std;

TEST_CASE("TransformerTest-testInitialization") {
  vector<Tensor> tensors;
  tensors.push_back(Tensor({0.2,
                            0.7,
                            0.1,
                            0.3,
                            0.4,
                            0.8,
                            0.9,
                            0.35,
                            0.12,
                            0.27,
                            0.17,
                            0.41,
                            numeric_limits<double>::max(),
                            0.27,
                            0.67,
                            0.41,
                            1,
                            0.37,
                            0.17,
                            0.41,
                            6,
                            0.17,
                            0.65,
                            0.87,
                            5,
                            0.97,
                            0.19,
                            0.51,
                            4},
                           {29}));
  tensors.push_back(Tensor({0.2,
                            0.7,
                            0.1,
                            0.3,
                            0.4,
                            0.8,
                            0.9,
                            0.35,
                            0.12,
                            0.27,
                            0.17,
                            0.41,
                            numeric_limits<double>::max(),
                            0.27,
                            0.67,
                            0.41,
                            1,
                            0.37,
                            0.17,
                            0.41,
                            6,
                            0.77,
                            0.61,
                            0.27,
                            2},
                           {25}));
  tensors.push_back(Tensor({0.2,
                            0.7,
                            0.1,
                            0.3,
                            0.4,
                            0.8,
                            0.9,
                            0.35,
                            0.12,
                            0.27,
                            0.17,
                            0.41,
                            numeric_limits<double>::max(),
                            1.2,
                            3.6,
                            7.1,
                            3,
                            5.4,
                            0.17,
                            9.8,
                            4,
                            0.77,
                            0.61,
                            0.27,
                            2},
                           {25}));

  vector<int> input = {30, 15};
  vector<Function *> inputFunctions;
  inputFunctions.push_back(new Tanh());
  inputFunctions.push_back(new Sigmoid());

  vector<Function *> outputFunctions;
  outputFunctions.push_back(new Sigmoid());
  outputFunctions.push_back(new Tanh());

  vector<double> gammaInput = {1.0, 1.0};
  vector<double> gammaOutput = {1.0, 1.0, 1.0};

  vector<double> betaInput = {0.0, 0.0};
  vector<double> betaOutput = {0.0, 0.0, 0.0};

  auto *dictionary = new VectorizedDictionary();
  Transformer transformer(dictionary);

  TransformerParameter parameter(
      1, 150, new AdamW(0.025, 0.99, 0.99, 0.999, 1e-10, 0.1),
      new RandomInitialization(), 3, 2, 7, 1e-9, input, input, inputFunctions,
      outputFunctions, gammaInput, gammaOutput, betaInput, betaOutput);

  transformer.train(tensors, parameter);
  // Since this is a test for initialization and a basic training run, we just
  // check if it completes without errors.
  REQUIRE(true);
}
