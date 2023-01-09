//
// Created by Olcay Taner YILDIZ on 9.01.2023.
//

#include "LabelledVectorizedWord.h"

LabelledVectorizedWord::LabelledVectorizedWord(const string &word, const Vector &vector, const string &classLabel) : VectorizedWord(word, vector) {
    this->classLabel = classLabel;
}

LabelledVectorizedWord::LabelledVectorizedWord(const string &word, const string &classLabel) : VectorizedWord(word, Vector(300, 0)){
    this->classLabel = classLabel;
}

string LabelledVectorizedWord::getClassLabel() const{
    return classLabel;
}
