import json
import os

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)

from tqdm.auto import tqdm

tqdm.pandas()

DATA_FOLDER = "~/Data/movie_search/MovieSummaries/"
character_file = "character.metadata.tsv"
movie_file = "movie.metadata.tsv"
name_file = "name.clusters.txt"
plot_file = "plot_summaries.txt"
output_file = "movie_plot_mapping.json"


def data_preprocessing():

    # read character data
    char_df_cols = [
        "Wikipedia_Movie_ID",
        "Freebase_Movie_ID",
        "Date",
        "Character_Name",
        "Actor_DOB",
        "Actor_Gender",
        "Actor_Height",
        "Actor_Ethnicity",
        "Actor_Name",
        "Actor_age_at_movie_release",
        "Freebase_character_map1",
        "Freebase_character_map2",
        "Freebase_character_map3",
    ]

    char_df = pd.read_csv(
        os.path.join(DATA_FOLDER, character_file),
        sep="\t",
        header=None,
        names=char_df_cols,
    )

    # read movie data
    movie_df_cols = [
        "Wikipedia_Movie_ID",
        "Freebase_Movie_ID",
        "Movie_Name",
        "Movie_release_date",
        "Movie_Revenue",
        "Movie_Runtime",
        "Movie_Language",
        "Movie_Country",
        "Movie_Genere",
    ]

    movie_df = pd.read_csv(
        (os.path.join(DATA_FOLDER, movie_file)),
        sep="\t",
        header=None,
        names=movie_df_cols,
    )

    # read charatcer names
    name_df_cols = ["Character_Name", "Character_ID"]
    name_df = pd.read_csv(
        os.path.join(DATA_FOLDER, name_file), sep="\t", header=None, names=name_df_cols
    )

    # read plot data
    plot_df_cols = ["Wikipedia_Movie_ID", "Plot"]
    plot_df = pd.read_csv(
        os.path.join(DATA_FOLDER, plot_file), sep="\t", header=None, names=plot_df_cols
    )
    
    plots_map = {}
    for i in range(plot_df.shape[0]):
        movie_id = plot_df["Wikipedia_Movie_ID"].iloc[i]
        plot = plot_df["Plot"].iloc[i]
        plots_map[movie_id] = plot

    # data imputation
    char_df = char_df.fillna("<blank>").copy()

    # add metadata
    title_desc_map = {}

    for idx in tqdm(range(movie_df.shape[0])):
        movie_name = movie_df["Movie_Name"].iloc[idx]
        wiki_movie_id = movie_df["Wikipedia_Movie_ID"].iloc[idx]
        if plots_map.get(wiki_movie_id, -1) != -1:
            title_desc_map[movie_name] = plots_map[wiki_movie_id]

    # save the movie to plot mapping
    with open(output_file, "w") as f:
        json.dump(title_desc_map, f)


if __name__ == "__main__":
    data_preprocessing()
