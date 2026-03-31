//
// Created by Olcay Taner YILDIZ on 9.01.2023.
//

#ifndef SEQUENCEPROCESSING_LABELLEDVECTORIZEDWORD_H
#define SEQUENCEPROCESSING_LABELLEDVECTORIZEDWORD_H


#include <Dictionary/VectorizedWord.h>

class LabelledVectorizedWord : public VectorizedWord{
private:
    string classLabel;
public:
    LabelledVectorizedWord(const string &word, const Vector &vector, const string &classLabel);
    LabelledVectorizedWord(const string& word, const string& classLabel);
    string getClassLabel() const;
};


#endif //SEQUENCEPROCESSING_LABELLEDVECTORIZEDWORD_H
