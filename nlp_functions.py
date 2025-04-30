import gensim
from nlp_project_functions import LCV, DV, TopWords

wv = gensim.models.KeyedVectors.load_word2vec_format('glove-wiki-gigaword-50.gz')
print(DV(wv, "dr. strange"))



from typing import List, Tuple, Union
import numpy as np
from numpy.linalg import norm
import nltk

# -------------------------------------------------
# 1. Keep lower-cased tokens that exist in the model
# -------------------------------------------------
def LCV(wv, LsWords: List[str] = ['NLP', 'is', 'fun', '!']) -> List[str]:
    return [tok for tok in map(str.lower, LsWords) if tok in wv]

# -------------------------------------------------
# 2. Document vector (centroid)
# -------------------------------------------------
def DV(wv, LsWords: Union[str, List[str]] = ['I', 'like', 'nlp']) -> np.ndarray:
    if isinstance(LsWords, str):          # leave raw string for char-wise LCV
        tokens = LCV(wv, LsWords)
    else:
        tokens = LCV(wv, LsWords)
    if not tokens:
        return np.zeros(wv.vector_size)
    return wv[tokens].mean(axis=0)

# -------------------------------------------------
# 3. Top-n words closest to the document vector
# -------------------------------------------------
def TopWords(
    wv,
    LsWords: List[str] = ['super', 'cow', 'now'],
    n: int = 1,
) -> Union[List[Tuple[str, float]], None]:
    lcv = LCV(wv, LsWords)
    if not lcv:
        return None
    dv       = DV(wv, LsWords)
    dv_norm  = norm(dv)
    uniq     = list(dict.fromkeys(lcv))       # remove dups, keep order
    mat      = wv[uniq]                       # (k, 50)
    sims     = (mat @ dv) / (np.linalg.norm(mat, axis=1) * dv_norm)
    top_idx  = sims.argsort()[::-1][:n]
    return [(uniq[i], float(sims[i])) for i in top_idx]

# -------------------------------------------------
# 4. Word vector nearest the origin
# -------------------------------------------------
def NN0(wv, LsWords=['Cat', 'in', 'the']) -> Union[Tuple[str, float], None]:
    tokens = LCV(wv, LsWords)
    if not tokens:
        return None
    mat   = wv[tokens]
    norms = np.linalg.norm(mat, axis=1)
    idx   = norms.argmin()
    return (tokens[idx], float(norms[idx]))

# -------------------------------------------------
# 5. Cumulative mean of a 1-D NumPy array
# -------------------------------------------------
def CumMean(x=np.array([0, 1, 2])) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.array([])
    return np.cumsum(arr) / np.arange(1, arr.size + 1)

# -------------------------------------------------
# 6. Pick the nPos-th coordinate from each word vector
# -------------------------------------------------
def PickPosValues(
    wv,
    LsWords: List[str] = ['NLP', 'is'],
    nPos: int = 0,
) -> List[float]:
    return [float(wv[w][nPos]) for w in LCV(wv, LsWords)]

# -------------------------------------------------
# 7. Cumulative mean of that coordinate, skipping first nSkip
# -------------------------------------------------
def CumMeanInPos(
    wv,
    LsWords: List[str] = ['Hi', 'from', 'Mars', '!'],
    nPos: int = 0,
    nSkip: int = 50,
) -> np.ndarray:
    vals = PickPosValues(wv, LsWords, nPos)
    if len(vals) <= nSkip:
        return np.array([])
    return CumMean(vals)[nSkip:]

# -------------------------------------------------
# 8. Nearest presidential inaugural speech
# -------------------------------------------------
def NN_Pres(
    wv,
    sQryFID: str = '1945-Roosevelt.txt',
    LvCorpusFID: List[str] = ['1865-Lincoln.txt', '1905-Roosevelt.txt'],
) -> Union[Tuple[str, float], None]:
    from nltk.corpus import inaugural

    if sQryFID not in inaugural.fileids():
        return None

    v_q   = DV(wv, inaugural.words(sQryFID))
    qnorm = norm(v_q)
    if qnorm == 0:
        return None

    cand_ids = [
        fid for fid in LvCorpusFID
        if fid != sQryFID and fid in inaugural.fileids()
    ]

    best_id, best_sim = None, -1.0
    for fid in cand_ids:
        v_c   = DV(wv, inaugural.words(fid))
        cnorm = norm(v_c)
        if cnorm == 0:
            continue
        sim = float(np.dot(v_q, v_c) / (qnorm * cnorm))
        if sim > best_sim:
            best_id, best_sim = fid, sim

    return None if best_id is None else (best_id, best_sim)
