# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
input_dataset_prepared = dataiku.Dataset("input_dataset_prepared")
input_dataset_prepared_df = input_dataset_prepared.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

embeddings_df = input_dataset_prepared_df # For this sample code, simply copy input to output


# Write recipe outputs
embeddings = dataiku.Dataset("embeddings")
embeddings.write_with_schema(embeddings_df)
import os
from sentence_transformers import SentenceTransformer

# Load pre-trained model
sentence_transformer_home = os.getenv('SENTENCE_TRANSFORMERS_HOME')
model_path = os.path.join(sentence_transformer_home, 'DataikuNLP_average_word_embeddings_glove.6B.300d')
model = SentenceTransformer(model_path)

sentences = ["I really like Ice cream", "Brussels sprouts are okay too"]

# get sentences embeddings
embeddings = model.encode(sentences)
embeddings.shape