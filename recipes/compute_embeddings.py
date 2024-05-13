# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import os
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import logging
logging.basicConfig(
    level=logging.WARNING
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
input_dataset_prepared = dataiku.Dataset("input_dataset_prepared")
df = input_dataset_prepared.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client=dataiku.api_client()
project=client.get_project(dataiku.default_project_key())
variables=project.get_variables()
vars=variables["standard"]
vars

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Load pre-trained model
sentence_transformer_home = os.getenv('SENTENCE_TRANSFORMERS_HOME')
model_path = os.path.join(sentence_transformer_home, vars["embedding_model"])
model = SentenceTransformer(model_path)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sentences=df["text"].to_list()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get sentences embeddings
embeddings = model.encode(sentences)
embeddings.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
embeddings = dataiku.Dataset("embeddings")
embeddings.write_with_schema(embeddings)