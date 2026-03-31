//
// Created by Olcay Taner YILDIZ on 9.01.2023.
//

#ifndef SEQUENCEPROCESSING_SEQUENCECORPUS_H
#define SEQUENCEPROCESSING_SEQUENCECORPUS_H


#include <Corpus.h>

class SequenceCorpus : public Corpus{
public:
    explicit SequenceCorpus(const string& fileName);
    vector<string> getClassLabels();
};


#endif //SEQUENCEPROCESSING_SEQUENCECORPUS_H
