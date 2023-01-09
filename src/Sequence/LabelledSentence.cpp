//
// Created by Olcay Taner YILDIZ on 9.01.2023.
//

#include "LabelledSentence.h"

LabelledSentence::LabelledSentence(const string &classLabel) : Sentence() {
    this->classLabel = classLabel;
}

string LabelledSentence::getClassLabel() const {
    return classLabel;
}
