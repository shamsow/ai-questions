import nltk
import sys
import string
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    # print(filenames)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    # print(matches)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data = {}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            data[filename] = f.read().replace("\n", " ")
    return data


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctuations = string.punctuation
    stopwords = nltk.corpus.stopwords.words("english")
    words = []
    for word in nltk.word_tokenize(document):
        if word not in punctuations and word not in stopwords:
            words.append(word.lower())
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    for filename in documents:
        words.update(documents[filename])

    # Calculate IDFs
    idfs = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Calculate TF-IDFs
    tfidfs = dict()
    for filename in files:
        tfidfs[filename] = 0
        for word in query:
            tf = files[filename].count(word)
            # tf = sum(word in files[filename]for i in files[filename])
            # tfidfs[filename].append((word, tf * idfs[word]))
            tfidfs[filename] += (tf * idfs[word])
    top = sorted(tfidfs, key=lambda score: tfidfs[score], reverse=True)
    return top[:n]
    

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scores = {}
    # print(sentences)
    for sentence in sentences:
        score = 0
        match_count = set()
        for word in query:
            if word in sentences[sentence]:
                score += idfs[word]
                match_count.add(word)
        if len(match_count) > 0:
            # print(len(sentences[sentence]), len(match_count))
            qt_density = len(match_count) / len(sentences[sentence])
            scores[sentence] = {"score": score, "density": qt_density}
    top = sorted(scores, key=lambda sentence: (scores[sentence]["score"], scores[sentence]["density"]), reverse=True)
    # print(top[0], scores[top[0]])
    # print(top[1], scores[top[1]])


    return top[:n]

            



if __name__ == "__main__":
    main()
