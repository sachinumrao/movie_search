# Build a streamlit app

import streamlit as st
from utils import MovieModel

def main():
    # load word2vec model
    model = MovieModel()

    # build input field
    
    # build search button
    
    # get scores for search query
    preds = model.score(search_phrase)
    # draw results 

if __name__ == "__main__":
    main()