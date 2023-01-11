//
// Created by Olcay Taner YILDIZ on 11.01.2023.
//

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "../src/Sequence/SequenceCorpus.h"

TEST_CASE("SequenceCorpusTest-testCorpus01") {
    SequenceCorpus corpus = SequenceCorpus("disambiguation-penn.txt");
    REQUIRE(25957 == corpus.sentenceCount());
    REQUIRE(264930 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus02") {
    SequenceCorpus corpus = SequenceCorpus("postag-atis-en.txt");
    REQUIRE(5432 == corpus.sentenceCount());
    REQUIRE(61879 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus03") {
    SequenceCorpus corpus = SequenceCorpus("slot-atis-en.txt");
    REQUIRE(5432 == corpus.sentenceCount());
    REQUIRE(61879 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus04") {
    SequenceCorpus corpus = SequenceCorpus("slot-atis-tr.txt");
    REQUIRE(5432 == corpus.sentenceCount());
    REQUIRE(45875 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus05") {
    SequenceCorpus corpus = SequenceCorpus("disambiguation-atis.txt");
    REQUIRE(5432 == corpus.sentenceCount());
    REQUIRE(45875 == corpus.numberOfWords());
}
