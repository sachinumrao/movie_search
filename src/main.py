# Build a streamlit app
import json

import numpy as np
import pandas as pd
import streamlit as st
from annoy import AnnoyIndex
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

INDEX_PATH = "/home/sachin/Data/movie_search/MovieSummaries/movie_vector_index.ann"
KEYMAP_PATH = "/home/sachin/Data/movie_search/MovieSummaries/movie_keymap.json"

VECTOR_LENGTH = 384
TOP_N = 5

ENCODER = SentenceTransformer("all-MiniLM-L6-v2")


def load_index():
    semantic_index = AnnoyIndex(VECTOR_LENGTH, "angular")
    semantic_index.load(INDEX_PATH)
    return semantic_index

def load_keymap():
    with open(KEYMAP_PATH, "r") as f0:
        keymap = json.load(f0)
    return keymap

def get_query_vector(query):
    sents = sent_tokenize(query)
    embeddings = ENCODER.encode(sents)
    pooled_embd = embeddings.mean(axis=0)
    return pooled_embd

def main():
    # load assets
    index = load_index()
    keymap = load_keymap()
    
    # heading
    st.title("Movie Search")
    
    # build input field
    query = st.text_input(label="Movie Description", value="")

    # build search button
    submit = st.button(label="Submit")

    # get scores for search query
    if submit:
        st.header("Top Results:")
        qvec = get_query_vector(query)
        neighbors = index.get_nns_by_vector(qvec, TOP_N, search_k=-1, include_distances=True)
        
        movies = pd.DataFrame({"Movie_ID": neighbors[0], "Score": neighbors[1]})
        movies["Movie"] = movies["Movie_ID"].apply(lambda x : keymap[str(x)])
        movies = movies.sort_values(by=["Score"]).copy()
        
        st.dataframe(data=movies)
        
        


if __name__ == "__main__":
    main()
