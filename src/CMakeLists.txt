cmake_minimum_required(VERSION 3.22)
project(SequenceProcessing)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_library(SequenceProcessing src/Sequence/LabelledSentence.h src/Sequence/LabelledVectorizedWord.cpp src/Sequence/LabelledVectorizedWord.h src/Sequence/LabelledSentence.cpp src/Sequence/LabelledSentence.h src/Sequence/SequenceCorpus.cpp src/Sequence/SequenceCorpus.h Test/SequenceCorpusTest.cpp)
add_executable(Test src/Sequence/LabelledSentence.h src/Sequence/LabelledVectorizedWord.cpp src/Sequence/LabelledVectorizedWord.h src/Sequence/LabelledSentence.cpp src/Sequence/LabelledSentence.h src/Sequence/SequenceCorpus.cpp src/Sequence/SequenceCorpus.h Test/SequenceCorpusTest.cpp)

target_link_libraries(SequenceProcessing ${CONAN_LIBS})
target_link_libraries(Test ${CONAN_LIBS})
