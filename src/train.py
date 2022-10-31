import json
import os

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

DATA_FOLDER = "~/Data/movie_search/MovieSummaries/"
movie_plot_file = (
    "/home/sachin/Data/movie_search/MovieSummaries/movie_plot_mapping.json"
)
output_file = "/home/sachin/Data/movie_search/MovieSummaries/movie_vector_mapping.npy"

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text):
    sents = sent_tokenize(text)
    embeddings = MODEL.encode(sents)
    pooled_embd = embeddings.mean(axis=0)
    return pooled_embd


def get_movie_vector_mapping(movie_map):
    vector_map = {}

    for key, val in tqdm(movie_map.items()):
        vec = get_embedding(val)
        vector_map[key] = vec

    return vector_map


def main():
    # read movie plot mapping
    with open(movie_plot_file, "r") as f:
        movie_map = json.load(f)

    # get movie vector mapping
    vector_map = get_movie_vector_mapping(movie_map)

    # save movie_vector mapping
    np.save(output_file, vector_map)


if __name__ == "__main__":
    # test_text = "I am going to scuba diving in Hawai. You can also join me."
    # embd = get_embedding(test_text)
    main()
