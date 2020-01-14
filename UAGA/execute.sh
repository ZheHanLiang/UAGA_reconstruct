#!/home/lzh/UAGA/sh

python initial_data_processing.py -d 0 -n 100 -s 0 -g 3 -r 0.1

python initial_data_processing.py -d 1 -n 100 -s 0 -g 3 -r 0.1

python initial_data_processing.py -d 2 -n 100 -s 0 -g 3 -r 0.1

deepwalk --format edgelist --input ./data/new_edge/lastfm_source_edges.txt --max-memory-data-size 0 --number-walks 80 --representation-size 32 --walk-length 40 --window-size 5 --workers 10 --output ./data/graph_embedding/lastfm_source.emb

deepwalk --format edgelist --input ./data/new_edge/lastfm_target_edges.txt --max-memory-data-size 0 --number-walks 80 --representation-size 32 --walk-length 40 --window-size 5 --workers 10 --output ./data/graph_embedding/lastfm_target.emb

deepwalk --format edgelist --input ./data/new_edge/flickr_source_edges.txt --max-memory-data-size 0 --number-walks 80 --representation-size 32 --walk-length 40 --window-size 5 --workers 10 --output ./data/graph_embedding/flickr_source.emb

deepwalk --format edgelist --input ./data/new_edge/flickr_target_edges.txt --max-memory-data-size 0 --number-walks 80 --representation-size 32 --walk-length 40 --window-size 5 --workers 10 --output ./data/graph_embedding/flickr_target.emb

deepwalk --format edgelist --input ./data/new_edge/myspace_source_edges.txt --max-memory-data-size 0 --number-walks 80 --representation-size 32 --walk-length 40 --window-size 5 --workers 10 --output ./data/graph_embedding/myspace_source.emb

deepwalk --format edgelist --input ./data/new_edge/myspace_target_edges.txt --max-memory-data-size 0 --number-walks 80 --representation-size 32 --walk-length 40 --window-size 5 --workers 10 --output ./data/graph_embedding/myspace_target.emb

python main.py
