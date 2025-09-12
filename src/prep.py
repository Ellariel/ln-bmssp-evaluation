import os, sys
import random
import numpy as np
import networkx as nx
from glob import glob
from networkx.utils.misc import graphs_equal
import json


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

from utils import *

base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, 
                                        "..", 
                                        "data"))

print('data_dir:', data_dir)

final_file = os.path.join(data_dir, 'prepared.json')

if os.path.isfile(final_file):
    os.remove(final_file)
if os.path.isfile(final_file + '.zip'):
    os.remove(final_file + '.zip')

file_list = [i for i in glob(os.path.join(data_dir, '*.zip'))]

data = {'graphs': {},
        'neighbours': {}}


for f in file_list:
    id = os.path.basename(f).split('.')[0]
    g = nx.read_gml(read_zip(f))
    components = [g.subgraph(c)
                    for c in sorted(nx.connected_components(g),
                        key=len, reverse=True)]
    g = components[0]
    g = nx.convert_node_labels_to_integers(g)

    random.seed(13)
    np.random.seed(13)

    for u, v in g.edges:
        g.edges[u, v]['fee_base_sat'] = max(float(g.edges[u, v]['fee_base_msat']), 1) / 1000
        g.edges[u, v]['fee_rate_sat'] = max(float(g.edges[u, v]['fee_proportional_millionths']), 1) / 10**6
        if 'htlc_maximum_msat' not in g.edges[u, v]:
            g.edges[u, v]['htlc_maximum_msat'] = random_amount() * 1000 + float(g.edges[u, v]['htlc_minimim_msat'])
        else:
            g.edges[u, v]['htlc_maximum_msat'] = float(g.edges[u, v]['htlc_maximum_msat'])
        g.edges[u, v]['capacity_sat'] = max(g.edges[u, v]['htlc_maximum_msat'], 1) / 1000
        g.edges[u, v]['delay'] = max(float(g.edges[u, v]['cltv_expiry_delta']), 1)
        g.edges[u, v]['weight'] = weight_function(g, u, v)

    weights = [g.edges[u, v]['weight'] 
                    for u, v in g.edges]
    assert len(set(weights)) == len(weights)

    n = {i: [(j, float(g.edges[i, j]['weight'])) 
                            for j in g.neighbors(i)] 
                                    for i in g.nodes}
    
    data['graphs'][id] = graph_to_dict(g)
    data['neighbours'][id] = n

    assert graphs_equal(graph_from_dict(data['graphs'][id]), g)


with open(final_file, 'w') as f:
    json.dump(data, f)
    f.flush()

write_zip(final_file)

if os.path.isfile(final_file):
    os.remove(final_file)

