# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, kmeans_plusplus

# Read recipe inputs
embeddings = dataiku.Dataset("embeddings")
embeddings_df = embeddings.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

clustering_df = embeddings_df # For this sample code, simply copy input to output


# Write recipe outputs
clustering = dataiku.Dataset("clustering")
clustering.write_with_schema(clustering_df)
