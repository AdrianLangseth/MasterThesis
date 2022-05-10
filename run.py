"""
Script for running on IDUN. Should do all preprocessing, training, testing.
"""

import os.path
import pandas as pd
import tokenizer_embedder
from content_preprocessing import script_prepare_data

# prepare article Data
article_data, cat_features_encoders, labels_class_weights, emb = script_prepare_data('home/lemeiz/content_refine')

# Prepare interaction data

# Prepare data sampling and generator

# build model with direct location

# build model with location interests

# Train & Test model with
