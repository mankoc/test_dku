# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.cluster import KMeans, kmeans_plusplus
import json

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
embeddings = dataiku.Dataset("embeddings")
df = embeddings.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client=dataiku.api_client()
project=client.get_project(dataiku.default_project_key())
variables=project.get_variables()
var=variables["standard"]
var

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
embeddings2=[json.loads(x) for x in df["embeddings"].to_list()]
embeddings=np.array(embeddings2)
embeddings.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
clustering = KMeans(n_clusters=var["n_clusters"],n_init='auto',init='k-means++').fit_predict(embeddings)
clustering

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
to_review = [False] * len(clustering)

for cl in np.unique(clustering):
    # print(len(np.where(clustering == cl)[0]))
    cluster_indexes = np.where(clustering == cl)[0]
    nelements = max(
        int(len(cluster_indexes) * var["select_ratio"]), var["min_instances"]
    )
    if nelements < var["min_instances"]:
        nelements = min(var["min_instances"], len(cluster_indexes))

    centers_init, indices = kmeans_plusplus(
        embeddings[cluster_indexes], n_clusters=nelements, random_state=0
    )
    # selected = np.random.choice(cluster_indexes, size=nelements)
    for ind in indices:
        to_review[ind] = True
    # to_review[indices]=True

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

clustering_df = embeddings_df # For this sample code, simply copy input to output


# Write recipe outputs
clustering = dataiku.Dataset("clustering")
clustering.write_with_schema(clustering_df)