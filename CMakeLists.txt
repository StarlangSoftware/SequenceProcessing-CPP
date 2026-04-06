cmake_minimum_required(VERSION 3.22)
project(SequenceProcessing)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

find_package(word_to_vec REQUIRED)
find_package(classification REQUIRED)
find_package(computational_graph REQUIRED)

add_library(SequenceProcessing
        src/Sequence/LabelledSentence.h src/Sequence/LabelledSentence.cpp
        src/Sequence/LabelledVectorizedWord.h src/Sequence/LabelledVectorizedWord.cpp
        src/Sequence/SequenceCorpus.h src/Sequence/SequenceCorpus.cpp
        src/Functions/AdditionByConstant.h src/Functions/AdditionByConstant.cpp
        src/Functions/Inverse.h src/Functions/Inverse.cpp
        src/Functions/Mask.h src/Functions/Mask.cpp
        src/Functions/Mean.h src/Functions/Mean.cpp
        src/Functions/MultiplyByConstant.h src/Functions/MultiplyByConstant.cpp
        src/Functions/RemoveBias.h src/Functions/RemoveBias.cpp
        src/Functions/SquareRoot.h src/Functions/SquareRoot.cpp
        src/Functions/Switch.h src/Functions/Switch.cpp
        src/Functions/Transpose.h src/Functions/Transpose.cpp
        src/Functions/Variance.h src/Functions/Variance.cpp
        src/Parameters/RecurrentNeuralNetworkParameter.h src/Parameters/RecurrentNeuralNetworkParameter.cpp
        src/Parameters/TransformerParameter.h src/Parameters/TransformerParameter.cpp
        src/Classification/RecurrentNeuralNetworkModel.h src/Classification/RecurrentNeuralNetworkModel.cpp
        src/Classification/GatedRecurrentUnitModel.h src/Classification/GatedRecurrentUnitModel.cpp
        src/Classification/LongShortTermMemoryModel.h src/Classification/LongShortTermMemoryModel.cpp
        src/Classification/Transformer.h src/Classification/Transformer.cpp)
target_link_libraries(SequenceProcessing word_to_vec::word_to_vec classification::classification computational_graph::computational_graph)

add_executable(Test
        src/Sequence/LabelledSentence.h src/Sequence/LabelledSentence.cpp
        src/Sequence/LabelledVectorizedWord.h src/Sequence/LabelledVectorizedWord.cpp
        src/Sequence/SequenceCorpus.h src/Sequence/SequenceCorpus.cpp
        src/Functions/AdditionByConstant.h src/Functions/AdditionByConstant.cpp
        src/Functions/Inverse.h src/Functions/Inverse.cpp
        src/Functions/Mask.h src/Functions/Mask.cpp
        src/Functions/Mean.h src/Functions/Mean.cpp
        src/Functions/MultiplyByConstant.h src/Functions/MultiplyByConstant.cpp
        src/Functions/RemoveBias.h src/Functions/RemoveBias.cpp
        src/Functions/SquareRoot.h src/Functions/SquareRoot.cpp
        src/Functions/Switch.h src/Functions/Switch.cpp
        src/Functions/Transpose.h src/Functions/Transpose.cpp
        src/Functions/Variance.h src/Functions/Variance.cpp
        src/Parameters/RecurrentNeuralNetworkParameter.h src/Parameters/RecurrentNeuralNetworkParameter.cpp
        src/Parameters/TransformerParameter.h src/Parameters/TransformerParameter.cpp
        src/Classification/RecurrentNeuralNetworkModel.h src/Classification/RecurrentNeuralNetworkModel.cpp
        src/Classification/GatedRecurrentUnitModel.h src/Classification/GatedRecurrentUnitModel.cpp
        src/Classification/LongShortTermMemoryModel.h src/Classification/LongShortTermMemoryModel.cpp
        src/Classification/Transformer.h src/Classification/Transformer.cpp
        Test/SequenceCorpusTest.cpp Test/TransformerTest.cpp)
target_link_libraries(Test word_to_vec::word_to_vec classification::classification computational_graph::computational_graph)

# Copy data .txt files to the build directory so tests can find them
file(GLOB DATA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.txt")
file(COPY ${DATA_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
