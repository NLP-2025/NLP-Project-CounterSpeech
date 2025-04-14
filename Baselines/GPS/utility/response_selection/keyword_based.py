import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
import warnings
from utility.response_selection import method


class BM25Method(method.BaselineMethod):
    """BM25 baseline, using weighted keyword matching.

    Adapted from https://github.com/arosh/BM25Transformer/blob/master/bm25.py
    see Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html

    Args:
        k1: float, optional (default=2.0)
        b: float, optional (default=0.75)
    """

    def __init__(self, k1=2.0, b=0.75):
        """Create a new `BM25Method` object."""
        self._k1 = k1
        self._b = b

    def train(self, contexts, responses):
        """Fit the tf-idf transform and compute idf statistics."""
        self._vectorizer = HashingVectorizer(non_negative=True)
        self._tfidf_transform = TfidfTransformer()

        # Create count matrix (term frequencies) for contexts and responses combined
        count_matrix = self._tfidf_transform.fit_transform(self._vectorizer.transform(contexts + responses))
        n_samples, n_features = count_matrix.shape

        # Compute document frequency
        df = np.diff(count_matrix.indptr)
        idf = np.log((n_samples - df + 0.5) / (df + 0.5))  # BM25-specific IDF formula

        # Create a sparse diagonal matrix of the IDF values
        self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)

        # Compute average document length
        document_lengths = count_matrix.sum(axis=1)
        self._average_document_length = np.mean(document_lengths)

    def _vectorize(self, strings):
        """Vectorize the given strings."""
        tf_idf_vectors = self._tfidf_transform.transform(self._vectorizer.transform(strings))
        tf_idf_vectors = sp.csr_matrix(tf_idf_vectors, dtype=np.float64, copy=True)

        # Compute document lengths
        document_lengths = tf_idf_vectors.sum(axis=1)

        # Number of terms (non-zero elements) in each document
        num_terms = tf_idf_vectors.indptr[1:] - tf_idf_vectors.indptr[:-1]

        # Repeat document lengths for each non-zero element
        rep = np.repeat(np.asarray(document_lengths), num_terms)

        # Compute BM25 score only for non-zero elements
        data = tf_idf_vectors.data * (self._k1 + 1) / (
            tf_idf_vectors.data + self._k1 * (1 - self._b + self._b * rep / self._average_document_length)
        )

        # Rebuild the sparse matrix with BM25 scores
        vectors = sp.csr_matrix((data, tf_idf_vectors.indices, tf_idf_vectors.indptr), shape=tf_idf_vectors.shape)
        vectors = vectors * self._idf_diag

        return vectors

    def cal_similarity(self, contexts, responses):
        contexts_matrix = self._vectorize(contexts)
        responses_matrix = self._vectorize(responses)
        similarities = contexts_matrix.dot(responses_matrix.T).toarray()
        return similarities

    def rank_responses(self, contexts, responses):
        """Rank the responses for each context."""
        similarities = self.cal_similarity(contexts, responses)
        return np.argmax(similarities, axis=1)

    def sort_responses(self, contexts, responses, num):
        """Sort and return the top `num` ranked responses for each context."""
        similarities = self.cal_similarity(contexts, responses)
        return [x[::-1][:num] for x in np.argsort(similarities, axis=1)]


class TfIdfMethod(method.BaselineMethod):
    """TF-IDF baseline.

    This hashes words to sparse IDs, and then computes tf-idf statistics for
    these hashed IDs. As a result, no words are considered out-of-vocabulary.
    """

    def train(self, contexts, responses):
        """Fit the tf-idf transform and compute idf statistics."""
        self._vectorizer = HashingVectorizer()
        self._tfidf_transform = TfidfTransformer()
        self._tfidf_transform.fit(self._vectorizer.transform(contexts + responses))

    def _vectorize(self, strings):
        """Vectorize the given strings."""
        tf_idf_vectors = self._tfidf_transform.transform(self._vectorizer.transform(strings))
        return sp.csr_matrix(tf_idf_vectors, dtype=np.float64, copy=True)

    def cal_similarity(self, contexts, responses):
        contexts_matrix = self._vectorize(contexts)
        responses_matrix = self._vectorize(responses)
        similarities = contexts_matrix.dot(responses_matrix.T).toarray()
        return similarities

    def rank_responses(self, contexts, responses):
        """Rank the responses for each context."""
        similarities = self.cal_similarity(contexts, responses)
        return np.argmax(similarities, axis=1)

    def sort_responses(self, contexts, responses, num):
        """Sort and return the top `num` ranked responses for each context."""
        similarities = self.cal_similarity(contexts, responses)
        return [x[::-1][:num] for x in np.argsort(similarities, axis=1)]
