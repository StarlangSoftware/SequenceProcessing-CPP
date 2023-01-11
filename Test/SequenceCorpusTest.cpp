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

TEST_CASE("SequenceCorpusTest-testCorpus06") {
    SequenceCorpus corpus = SequenceCorpus("metamorpheme-atis.txt");
    REQUIRE(5432 == corpus.sentenceCount());
    REQUIRE(45875 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus07") {
    SequenceCorpus corpus = SequenceCorpus("postag-atis-tr.txt");
    REQUIRE(5432 == corpus.sentenceCount());
    REQUIRE(45875 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus08") {
    SequenceCorpus corpus = SequenceCorpus("metamorpheme-penn.txt");
    REQUIRE(25957 == corpus.sentenceCount());
    REQUIRE(264930 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus09") {
    SequenceCorpus corpus = SequenceCorpus("ner-penn.txt");
    REQUIRE(19118 == corpus.sentenceCount());
    REQUIRE(168654 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus10") {
    SequenceCorpus corpus = SequenceCorpus("postag-penn.txt");
    REQUIRE(25957 == corpus.sentenceCount());
    REQUIRE(264930 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus11") {
    SequenceCorpus corpus = SequenceCorpus("semanticrolelabeling-penn.txt");
    REQUIRE(19118 == corpus.sentenceCount());
    REQUIRE(168654 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus12") {
    SequenceCorpus corpus = SequenceCorpus("semantics-penn.txt");
    REQUIRE(19118 == corpus.sentenceCount());
    REQUIRE(168654 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus13") {
    SequenceCorpus corpus = SequenceCorpus("shallowparse-penn.txt");
    REQUIRE(9557 == corpus.sentenceCount());
    REQUIRE(87279 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus14") {
    SequenceCorpus corpus = SequenceCorpus("disambiguation-tourism.txt");
    REQUIRE(19830 == corpus.sentenceCount());
    REQUIRE(91152 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus15") {
    SequenceCorpus corpus = SequenceCorpus("metamorpheme-tourism.txt");
    REQUIRE(19830 == corpus.sentenceCount());
    REQUIRE(91152 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus16") {
    SequenceCorpus corpus = SequenceCorpus("postag-tourism.txt");
    REQUIRE(19830 == corpus.sentenceCount());
    REQUIRE(91152 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus17") {
    SequenceCorpus corpus = SequenceCorpus("semantics-tourism.txt");
    REQUIRE(19830 == corpus.sentenceCount());
    REQUIRE(91152 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus18") {
    SequenceCorpus corpus = SequenceCorpus("shallowparse-tourism.txt");
    REQUIRE(19830 == corpus.sentenceCount());
    REQUIRE(91152 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus19") {
    SequenceCorpus corpus = SequenceCorpus("disambiguation-kenet.txt");
    REQUIRE(18687 == corpus.sentenceCount());
    REQUIRE(178658 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus20") {
    SequenceCorpus corpus = SequenceCorpus("metamorpheme-kenet.txt");
    REQUIRE(18687 == corpus.sentenceCount());
    REQUIRE(178658 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus21") {
    SequenceCorpus corpus = SequenceCorpus("postag-kenet.txt");
    REQUIRE(18687 == corpus.sentenceCount());
    REQUIRE(178658 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus22") {
    SequenceCorpus corpus = SequenceCorpus("disambiguation-framenet.txt");
    REQUIRE(2704 == corpus.sentenceCount());
    REQUIRE(19286 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus23") {
    SequenceCorpus corpus = SequenceCorpus("metamorpheme-framenet.txt");
    REQUIRE(2704 == corpus.sentenceCount());
    REQUIRE(19286 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus24") {
    SequenceCorpus corpus = SequenceCorpus("postag-framenet.txt");
    REQUIRE(2704 == corpus.sentenceCount());
    REQUIRE(19286 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus25") {
    SequenceCorpus corpus = SequenceCorpus("semanticrolelabeling-framenet.txt");
    REQUIRE(2704 == corpus.sentenceCount());
    REQUIRE(19286 == corpus.numberOfWords());
}

TEST_CASE("SequenceCorpusTest-testCorpus26") {
    SequenceCorpus corpus = SequenceCorpus("sentiment-tourism.txt");
    REQUIRE(19830 == corpus.sentenceCount());
    REQUIRE(91152 == corpus.numberOfWords());
}
