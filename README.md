## 🛠️ Function List

### Sparse_Doc_Vectors

### `GetDTM()`
Generates a count-based Document-Term Matrix (DTM) using `CountVectorizer`.
### `GetFrac()`
Calculates the fraction of elements in the DTM matching a given value (e.g., sparsity).
### `MostCommonWords1()`
Finds the top n most frequent words by total count.
### `MostCommonWords2()`
Identifies the most common words based on document occurrence, not repetition.
### `MostCommonWords3()`
Selects words with the highest single-document frequency.
### `SentWithMostDups()`
Returns the sentence with the most repeated instances of a given word.
### `GetTFIDF()`
Transforms a count-based DTM into a TF-IDF weighted matrix.
### `MostImportantWords()`
Ranks words by their highest TF-IDF scores.
### `LeastImportantWords()`
Identifies low-weighted words (auto stopwords) based on minimum non-zero TF-IDF values.

### Dense_Doc_Vectors

### GetWV()
Retrieves the word vector for a given input word (lowercased). If the word is not found in the Word2Vec vocabulary, returns a zero vector of the correct length.
### GetSupWords()
Searches the Word2Vec vocabulary for words that contain the given input word as a substring (case-insensitive). Returns all matching vocabulary words.
### NN()
Finds the most similar neighbors for a given word, filtered by a similarity threshold. Returns only neighbors with a similarity score above the specified threshold.
### NN2()
Identifies the single pair of words within a given list that are the most semantically similar, based on cosine similarity. Returns the pair with the highest score, sorted alphabetically.
### NNExc()
Finds the most similar neighbor to a given word while excluding any words from a provided exception list. Useful for avoiding duplicates or unwanted terms in similarity searches.
### NNChain()
Builds a sequential chain of semantically similar words. Starts with a given input word and adds the closest neighbor at each step, excluding all previously selected words. The chain continues until the specified length is reached or no valid neighbors remain.

### K-NN   (K Nearest Neighbors)

### findknn()
Finds the indices and distances of the k-nearest neighbors in the training set (xTr) for each test point (xTe). Returns both the neighbor indices and their corresponding distances.
### accuracy()
Calculates the classification accuracy as the fraction of correct predictions. Compares the predicted labels (preds) to the true labels (truth).
### knnclassifier()
Implements the k-Nearest Neighbors (k-NN) classifier. Uses the neighbor indices from findknn() and applies majority voting (mode) to predict labels for each test point.
### findknn_test1()
Verifies that findknn() returns the correct data types (integer indices and float distances).
### findknn_test2()
Confirms that findknn() outputs matrices with the correct shapes based on the input sizes and chosen k.
### findknn_test3()
Validates the accuracy of findknn() for the 1-nearest neighbor case by comparing against known correct results.
### findknn_test4()
Checks the correctness of findknn() for the 3-nearest neighbor scenario, ensuring alignment with expected output.
