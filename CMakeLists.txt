cmake_minimum_required(VERSION 3.22)
project(SequenceProcessing)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

find_package(word_to_vec REQUIRED)
find_package(classification REQUIRED)

add_library(SequenceProcessing src/Sequence/LabelledSentence.h src/Sequence/LabelledVectorizedWord.cpp src/Sequence/LabelledVectorizedWord.h src/Sequence/LabelledSentence.cpp src/Sequence/LabelledSentence.h src/Sequence/SequenceCorpus.cpp src/Sequence/SequenceCorpus.h Test/SequenceCorpusTest.cpp)
target_link_libraries(SequenceProcessing word_to_vec::word_to_vec classification::classification)
add_executable(Test src/Sequence/LabelledSentence.h src/Sequence/LabelledVectorizedWord.cpp src/Sequence/LabelledVectorizedWord.h src/Sequence/LabelledSentence.cpp src/Sequence/LabelledSentence.h src/Sequence/SequenceCorpus.cpp src/Sequence/SequenceCorpus.h Test/SequenceCorpusTest.cpp)
target_link_libraries(Test word_to_vec::word_to_vec classification::classification)
