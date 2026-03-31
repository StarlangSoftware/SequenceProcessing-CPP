cmake_minimum_required(VERSION 3.22)
project(SequenceProcessing)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

find_package(word_to_vec REQUIRED)
find_package(classification REQUIRED)
find_package(computational_graph REQUIRED)

add_library(SequenceProcessing
        Sequence/LabelledSentence.h Sequence/LabelledSentence.cpp
        Sequence/LabelledVectorizedWord.h Sequence/LabelledVectorizedWord.cpp
        Sequence/SequenceCorpus.h Sequence/SequenceCorpus.cpp
        Functions/AdditionByConstant.h Functions/AdditionByConstant.cpp
        Functions/Inverse.h Functions/Inverse.cpp
        Functions/Mask.h Functions/Mask.cpp
        Functions/Mean.h Functions/Mean.cpp
        Functions/MultiplyByConstant.h Functions/MultiplyByConstant.cpp
        Functions/RemoveBias.h Functions/RemoveBias.cpp
        Functions/SquareRoot.h Functions/SquareRoot.cpp
        Functions/Switch.h Functions/Switch.cpp
        Functions/Transpose.h Functions/Transpose.cpp
        Functions/Variance.h Functions/Variance.cpp
        Parameters/RecurrentNeuralNetworkParameter.h Parameters/RecurrentNeuralNetworkParameter.cpp
        Parameters/TransformerParameter.h Parameters/TransformerParameter.cpp
        Classification/RecurrentNeuralNetworkModel.h Classification/RecurrentNeuralNetworkModel.cpp
        Classification/GatedRecurrentUnitModel.h Classification/GatedRecurrentUnitModel.cpp
        Classification/LongShortTermMemoryModel.h Classification/LongShortTermMemoryModel.cpp
        Classification/Transformer.h Classification/Transformer.cpp)
target_link_libraries(SequenceProcessing word_to_vec::word_to_vec classification::classification computational_graph::computational_graph)
