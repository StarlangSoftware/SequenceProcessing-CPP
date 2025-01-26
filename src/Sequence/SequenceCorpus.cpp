//
// Created by Olcay Taner YILDIZ on 9.01.2023.
//

#include "SequenceCorpus.h"
#include "Dictionary/VectorizedWord.h"
#include "LabelledSentence.h"
#include "LabelledVectorizedWord.h"
#include "StringUtils.h"

SequenceCorpus::SequenceCorpus(const string &fileName) : Corpus() {
    string line, word;
    VectorizedWord* newWord;
    Sentence* newSentence = nullptr;
    ifstream inputStream;
    inputStream.open(fileName, ifstream::in);
    while (inputStream.good()){
        getline(inputStream, line);
        vector<string> items = StringUtils::split(line);
        word = items[0];
        if (word == "<S>") {
            if (items.size() == 2){
                newSentence = new LabelledSentence(items[1]);
            } else {
                newSentence = new Sentence();
            }
        } else {
            if (word == "</S>") {
                addSentence(newSentence);
            } else {
                if (items.size() == 2) {
                    newWord = new LabelledVectorizedWord(word, items[1]);
                } else {
                    newWord = new VectorizedWord(word, Vector(300,0));
                }
                if (newSentence != nullptr){
                    newSentence->addWord(newWord);
                }
            }
        }
    }
    inputStream.close();
}

vector<string> SequenceCorpus::getClassLabels() {
    bool sentenceLabelled = false;
    vector<string> classLabels;
    auto* t = (LabelledSentence*) sentences[0];
    if (t != nullptr){
        sentenceLabelled = true;
    }
    for (int i = 0; i < sentenceCount(); i++) {
        if (sentenceLabelled){
            auto* sentence = (LabelledSentence*) sentences[i];
            if (std::find(classLabels.begin(), classLabels.end(), sentence->getClassLabel()) == classLabels.end()) {
                classLabels.emplace_back(sentence->getClassLabel());
            }
        } else {
            Sentence* sentence = sentences[i];
            for (int j = 0; j < sentence->wordCount(); j++){
                auto* word = (LabelledVectorizedWord*) sentence->getWord(j);
                if (std::find(classLabels.begin(), classLabels.end(), word->getClassLabel()) == classLabels.end()) {
                    classLabels.emplace_back(word->getClassLabel());
                }
            }
        }
    }
    return classLabels;
}
