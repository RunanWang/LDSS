import pandas as pd
from pathlib import Path

data_root = Path("./job-light")

title_column_name = ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id",
                     "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"]
path = data_root / "title.csv"
df_title = pd.read_csv(path, header=0, sep=",", quotechar='"', escapechar="\\", low_memory=False)
df_title.columns = title_column_name
title_use_cols = ["id", "kind_id", "production_year", "episode_of_id", "season_nr", "episode_nr"]
df_title_light = df_title[title_use_cols]
# print(df_title_light.head(10))
# print(df_title_light.describe())
df_title_light.to_csv(path, index=False)
print("Table title processed.")

mi_column_name = ["id", "movie_id", "info_type_id", "info", "note"]
path = data_root / "movie_info.csv"
df_mi = pd.read_csv(path, header=0, sep=",", quotechar='"', escapechar="\\", low_memory=False)
df_mi.columns = mi_column_name
mi_use_cols = ["id", "movie_id", "info_type_id"]
df_mi_light = df_mi[mi_use_cols]
df_mi_light.to_csv(path, index=False)
print("Table movie_info processed.")

ci_column_name = ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"]
path = data_root / "cast_info.csv"
df_ci = pd.read_csv(path, header=0, sep=",", quotechar='"', escapechar="\\", low_memory=False)
df_ci.columns = ci_column_name
ci_use_cols = ["id", "person_id", "movie_id", "person_role_id", "nr_order", "role_id"]
df_ci_light = df_ci[ci_use_cols]
df_ci_light.to_csv(path, index=False)
print("Table cast_info processed.")

mc_column_name = ["id", "movie_id", "company_id", "company_type_id", "note"]
path = data_root / "movie_companies.csv"
df_mc = pd.read_csv(path, header=0, sep=",", quotechar='"', escapechar="\\", low_memory=False)
df_mc.columns = mc_column_name
mc_use_cols = ["id", "movie_id", "company_id", "company_type_id"]
df_mc_light = df_mc[mc_use_cols]
df_mc_light.to_csv(path, index=False)
print("Table movie_companies processed.")

mk_column_name = ["id", "movie_id", "keyword_id"]
path = data_root / "movie_keyword.csv"
df_mk = pd.read_csv(path, header=0, sep=",", quotechar='"', escapechar="\\", low_memory=False)
df_mk.columns = mk_column_name
df_mk.to_csv(path, index=False)
print("Table movie_keyword processed.")

miidx_column_name = ["id", "movie_id", "info_type_id", "info", "note"]
path = data_root / "movie_info_idx.csv"
df_miidx = pd.read_csv(path, header=0, sep=",", quotechar='"', escapechar="\\", low_memory=False)
df_miidx.columns = miidx_column_name
miidx_use_cols = ["id", "movie_id", "info_type_id"]
df_miidx_light = df_miidx[miidx_use_cols]
df_miidx_light.to_csv(path, index=False)
print("Table movie_info_idx processed.")
