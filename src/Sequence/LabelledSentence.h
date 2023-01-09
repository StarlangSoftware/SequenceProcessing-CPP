//
// Created by Olcay Taner YILDIZ on 9.01.2023.
//

#ifndef SEQUENCEPROCESSING_LABELLEDSENTENCE_H
#define SEQUENCEPROCESSING_LABELLEDSENTENCE_H


#include <Sentence.h>

class LabelledSentence : public Sentence{
private:
    string classLabel;
public:
    explicit LabelledSentence(const string& classLabel);
    string getClassLabel() const;
};


#endif //SEQUENCEPROCESSING_LABELLEDSENTENCE_H
