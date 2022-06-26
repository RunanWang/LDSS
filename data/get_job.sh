# wget http://homepages.cwi.nl/~boncz/job/imdb.tgz

mkdir job-light

tar -zxvf imdb.tgz -C job-light/

cd job-light

rm aka_name.csv aka_title.csv char_name.csv company_name.csv company_type.csv comp_cast_type.csv complete_cast.csv info_type.csv keyword.csv kind_type.csv link_type.csv movie_link.csv name.csv person_info.csv role_type.csv schematext.sql

cd ..

python process.py