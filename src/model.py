import json

import joblib
import numpy as np
from annoy import AnnoyIndex
from tqdm.auto import tqdm

input_file = "/home/sachin/Data/movie_search/MovieSummaries/movie_vector_mapping.pkl"
index_output_file = (
    "/home/sachin/Data/movie_search/MovieSummaries/movie_vector_index.ann"
)
keymap_output_file = "/home/sachin/Data/movie_search/MovieSummaries/movie_keymap.json"

VECTOR_LENGTH = 384


def main():
    # load numpy vectors
    movie_vectors = joblib.load(input_file)

    # create idx to movie
    idx2vec = {}
    idx2mov = {}

    idx = 0
    for key, val in movie_vectors.items():
        idx2vec[idx] = val
        idx2mov[idx] = key
        idx += 1

    # create index
    semantic_index = AnnoyIndex(VECTOR_LENGTH, "angular")

    # populate index
    for idx, vec in tqdm(idx2vec.items()):
        semantic_index.add_item(idx, vec)

    # save index
    semantic_index.build(128)
    semantic_index.save(index_output_file)

    # save keymap file
    with open(keymap_output_file, "w") as f0:
        json.dump(idx2mov, f0)


if __name__ == "__main__":
    main()
