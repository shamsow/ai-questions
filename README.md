# AI Questions

This AI uses the `nltk` package to tokenize the significant words in all the provided text files.

The words are assigned a score using the tf-idf methodology.

A question is received as input.

These scores are then used with the query to first:
- Rank the files in order of importance to the query.
- Rank the sentences in the top files according to importance to the query.
    - In case of a tie between sentences, the sentence which contains the larger query term density is ranked higher.

The top sentence is then returned as the answer to the question.