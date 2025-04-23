'''
word2vec_neighbors.py

Author: Dr. Benjamin Barnhill
Description:
This script provides helper functions for working with Word2Vec models using Gensim.
The functions support retrieving word vectors, finding similar neighbors, excluding specific words,
and building semantic neighbor chains. These tools are useful for tasks like semantic analysis,
keyword clustering, and exploratory analysis of word embeddings.

Use Case Example:
- Building word association chains for NLP projects.
- Analyzing semantic similarity between terms.
- Filtering similar terms while excluding certain words (e.g., for data cleaning or modeling).

Model Compatibility:
- Designed for use with pre-trained Gensim Word2Vec models like `glove-wiki-gigaword-50`.
'''

from itertools import combinations

# Function 1: Retrieve a Word Vector or Zero Vector
def GetWV(wv, sWord='nlp') -> np.array:
    '''Return the vector for sWord (lowercase), or a zero vector if not found.'''
    sWord = sWord.lower()
    return wv[sWord] if sWord in wv else np.zeros(wv.vector_size)

# Function 2: Extract Qualifying Words (Substring Matching)
def GetSupWords(wv, sWord='nlp') -> [str]:
    '''Return all vocabulary words where sWord (lowercase) is a substring.'''
    sWord = sWord.lower()
    return [word for word in wv.key_to_index if sWord in word]

# Function 3: Find Nearest Neighbors Above Similarity Threshold
def NN(wv, sWord='pet', nThreshold=0.75) -> [(str, float)]:
    '''Return a list of neighbors with similarity scores above nThreshold.'''
    sWord = sWord.lower()
    if sWord not in wv:
        return []
    return [(w, s) for w, s in wv.most_similar(sWord, topn=len(wv.key_to_index)) if s > nThreshold]

# Function 4: Find the Closest Pair of Words in a List
def NN2(wv, LsWords=['cat', 'dog', 'nlp']) -> (float, str, str):
    '''Identify the pair of closest words (highest similarity) in LsWords.'''
    valid_words = [word.lower() for word in LsWords if word.lower() in wv]
    if len(valid_words) < 2:
        return None
    best_score = -1
    best_pair = (None, None)
    for w1, w2 in combinations(valid_words, 2):
        score = wv.similarity(w1, w2)
        if score > best_score:
            best_score = score
            best_pair = (w1, w2)
    return (best_score, *sorted(best_pair))

# Function 5: Find Neighbors Excluding Certain Words
def NNExc(wv, sWord='pet', LsExcept=['cat', 'dog']) -> (str, float):
    '''Find the most similar neighbor to sWord excluding words in LsExcept.'''
    sWord = sWord.lower()
    LsExcept = [word.lower() for word in LsExcept]
    if sWord not in wv:
        return None
    neighbors = wv.most_similar(sWord, topn=len(wv.key_to_index))
    for word, score in neighbors:
        if word not in LsExcept:
            return (word, score)
    return None

# Function 6: Build a Chain of Neighbors
def NNChain(wv, sWord='pet', n=5) -> [(str, float)]:
    '''Build a chain of n neighbors, each closest to the previous word, excluding prior selections.'''
    sWord = sWord.lower()
    if sWord not in wv:
        return []
    chain = []
    used_words = [sWord]
    current_word = sWord
    for _ in range(n):
        next_neighbor = NNExc(wv, current_word, used_words)
        if next_neighbor is None:
            break
        chain.append(next_neighbor)
        used_words.append(next_neighbor[0])
        current_word = next_neighbor[0]
    return chain
