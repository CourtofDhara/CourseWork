## 🛠️ Function List

<details>
<summary>📂 Sparse Document Vectors</summary>

| Function               | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| `GetDTM()`              | Creates a count-based Document-Term Matrix (DTM) using `CountVectorizer`. |
| `GetFrac()`             | Calculates the fraction of elements matching a given value (e.g., sparsity). |
| `MostCommonWords1()`    | Finds the top n most frequent words by total count.            |
| `MostCommonWords2()`    | Identifies the most common words based on document occurrence, not repetition. |
| `MostCommonWords3()`    | Selects words with the highest single-document frequency.      |
| `SentWithMostDups()`    | Returns the sentence with the most repeated instances of a given word. |
| `GetTFIDF()`            | Converts a count-based DTM into a TF-IDF weighted matrix.      |
| `MostImportantWords()`  | Ranks words by their highest TF-IDF scores.                   |
| `LeastImportantWords()` | Identifies low-weighted words (auto stopwords) based on minimum non-zero TF-IDF values. |

</details>

---

<details>
<summary>📂 Dense Document Vectors</summary>

| Function           | Purpose                                                           |
|--------------------|-------------------------------------------------------------------|
| `GetWV()`          | Retrieves the Word2Vec vector for a given word (lowercased). Returns a zero vector if the word is not found. |
| `GetSupWords()`    | Finds all vocabulary words containing the given substring (case-insensitive). |
| `NN()`             | Finds the nearest neighbors for a given word above a similarity threshold. |
| `NN2()`            | Identifies the most similar pair of words in a list, based on cosine similarity. |
| `NNExc()`          | Returns the closest neighbor to a word, excluding specified exception terms. |
| `NNChain()`        | Builds a chain of semantically similar words, adding one closest neighbor at a time until the chain reaches the target length. |

</details>

---

<details>
<summary>📂 K-Nearest Neighbors (k-NN)</summary>

| Function             | Purpose                                                         |
|-----------------------|-----------------------------------------------------------------|
| `findknn()`           | Finds indices and distances of k-nearest neighbors between training (`xTr`) and test (`xTe`) sets. |
| `accuracy()`          | Calculates classification accuracy by comparing predicted vs. true labels. |
| `knnclassifier()`     | Classifies test points using k-NN majority voting.             |
| `findknn_test1()`     | Verifies that `findknn()` returns the correct data types.      |
| `findknn_test2()`     | Confirms that output shapes from `findknn()` match expectations. |
| `findknn_test3()`     | Validates 1-nearest neighbor accuracy against known results.   |
| `findknn_test4()`     | Checks correctness of the 3-nearest neighbor scenario.         |

</details>

<details>
<summary>📂 Document-Vector Similarity Toolkit</summary>

| Function             | Purpose |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `LCV()`              | Returns lower-cased tokens that exist in the loaded GloVe model—prevents key errors and shrinks memory use. |
| `DV()`               | Builds a **document vector** by averaging all word vectors that survive `LCV()`. Zero vector if none survive. |
| `TopWords()`         | For a given document, finds the *n* unique words whose vectors are most cosine-similar to the document vector. |
| `NN0()`              | Among candidate tokens, returns the word whose vector lies **closest to the origin** (smallest Euclidean norm). |
| `CumMean()`          | Computes the cumulative (running) mean of a 1-D NumPy array without loops. |
| `PickPosValues()`    | Pulls the value at a specified coordinate (`nPos`) from each word vector in a token list. |
| `CumMeanInPos()`     | Builds a cumulative mean time-series for a chosen vector coordinate, optionally skipping the first `nSkip` (volatile) points. |
| `NN_Pres()`          | Finds the U.S. inaugural speech **most similar** (cosine similarity) to a query speech using the centroid method above. |

</details>
