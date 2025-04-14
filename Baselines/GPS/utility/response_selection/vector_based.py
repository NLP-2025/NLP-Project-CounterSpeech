import abc
import itertools
import shutil
import tempfile
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glog
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utility.response_selection import method

class Encoder(abc.ABC):
    """A model that maps from text to dense vectors."""
    @abc.abstractmethod
    def encode_context(self, contexts):
        pass

    def encode_response(self, responses):
        """Encode the given response texts as vectors."""
        return self.encode_context(responses)

class TfHubEncoder:
    def __init__(self, model_url):
        # Load the model
        model = hub.load(model_url)
        # Extract the callable signature
        self.embed_fn = model if callable(model) else model.signatures['default']
    
    def encode_context(self, contexts):
        # Ensure contexts are properly formatted (list of strings)
        return self.embed_fn(contexts).numpy()
    def encode_response(self,responses):
        return self.embed_fn(responses).numpy()

class VectorSimilarityMethod(method.BaselineMethod):
    """Ranks responses using cosine similarity of context & response vectors."""
    def __init__(self, encoder):
        """Create a new `VectorSimilarityMethod` object."""
        self._encoder = encoder

    def train(self, contexts, responses):
        """Train on the contexts and responses. Does nothing."""
        pass

    def rank_responses(self, contexts, responses):
        """Rank the responses for each context, using cosine similarity."""
        contexts_matrix = self._encoder.encode_context(contexts)
        responses_matrix = self._encoder.encode_response(responses)
        
        # Normalize the response vectors
        responses_matrix /= np.linalg.norm(responses_matrix, axis=1, keepdims=True)
        
        # Compute similarities via dot product
        similarities = np.matmul(contexts_matrix, responses_matrix.T)
        return np.argmax(similarities, axis=1)

class VectorMappingMethod(method.BaselineMethod):
    """Applies a linear mapping to the response side and ranks with similarity."""
    def __init__(self, encoder, learning_rates=(10.0, 3.0, 1.0, 0.3, 0.01), regularizers=(0, 0.1, 0.01, 0.001)):
        """Create a new `VectorMappingMethod` object."""
        self._encoder = encoder
        self._learning_rates = learning_rates
        self._regularizers = regularizers
        self._build_model()

    def _build_model(self):
        """Build the model and graph for vector mapping."""
        self.contexts_placeholder = tf.keras.Input(shape=(None,), dtype=tf.float32)
        self.responses_placeholder = tf.keras.Input(shape=(None,), dtype=tf.float32)

        context_embeddings = self.contexts_placeholder
        response_embeddings = self.responses_placeholder

        # Normalize vectors
        context_embeddings = tf.nn.l2_normalize(context_embeddings, axis=-1)
        response_embeddings = tf.nn.l2_normalize(response_embeddings, axis=-1)

        # Apply mapping
        mapping_weights = tf.Variable(tf.random_normal_initializer()(shape=[response_embeddings.shape[-1], response_embeddings.shape[-1]]))
        response_mapped = tf.matmul(response_embeddings, mapping_weights)

        # Compute similarity
        similarities = tf.matmul(context_embeddings, response_mapped, transpose_b=True)

        self.model = tf.keras.Model(inputs=[self.contexts_placeholder, self.responses_placeholder], outputs=similarities)

    def train(self, contexts, responses):
        """Train the vector mapping model."""
        # Prepare training and dev sets
        contexts_train, contexts_dev, responses_train, responses_dev = self._create_train_and_dev(contexts, responses)
        
        # Training loop (could be a simple fit using Keras)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit([contexts_train, responses_train], responses_train, epochs=10, batch_size=32)

    def _create_train_and_dev(self, contexts, responses):
        """Create a train and dev set of context and response vectors."""
        context_encodings = np.array([self._encoder.encode_context([context])[0] for context in contexts])
        response_encodings = np.array([self._encoder.encode_response([response])[0] for response in responses])
        return train_test_split(context_encodings, response_encodings, test_size=0.2)

    def rank_responses(self, contexts, responses):
        """Rank the responses for each context."""
        context_encodings = np.array([self._encoder.encode_context([context])[0] for context in contexts])
        response_encodings = np.array([self._encoder.encode_response([response])[0] for response in responses])

        similarities = self.model.predict([context_encodings, response_encodings])
        return np.argmax(similarities, axis=1)
