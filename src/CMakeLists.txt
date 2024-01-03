cmake_minimum_required(VERSION 3.22)
project(SequenceProcessing)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

find_package(word_to_vec REQUIRED)
find_package(classification REQUIRED)

add_library(SequenceProcessing Sequence/LabelledSentence.h Sequence/LabelledVectorizedWord.cpp Sequence/LabelledVectorizedWord.h Sequence/LabelledSentence.cpp Sequence/LabelledSentence.h Sequence/SequenceCorpus.cpp Sequence/SequenceCorpus.h)
target_link_libraries(SequenceProcessing word_to_vec::word_to_vec classification::classification)
