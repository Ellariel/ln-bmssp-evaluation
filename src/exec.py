import os, sys
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
import json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

from utils import *
from proto import *

base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, 
                                        "..", 
                                        "data"))
results_dir = os.path.abspath(os.path.join(base_dir, 
                                        "..", 
                                        "results"))

os.makedirs(results_dir, exist_ok=True)
print('data_dir:', data_dir)
print('results_dir:', results_dir)

data = json.load(read_zip(os.path.join(data_dir, 'prepared.json.zip')))

random.seed(13)
np.random.seed(13)

N = 1
block_capacity = 512
batch_size = 16

results = {}
for id in tqdm(data['graphs'].keys(), leave=False):
    g = data['graphs'][id]
    g = graph_from_dict(g)
    n = data['neighbours'][id]
    n = {int(k): v for k, v in n.items()}

    assert len(g.nodes) == len(n)

    results.setdefault(id, {'dijkstra': [],
                            'dijkstrax': [],
                            'bmssp': [],})
    for _ in range(N):
        i = np.random.randint(len(n))
        #i = i + 1 if i % 2 != 0 else i
        dijkstrax = get_runtime(dijkstrax_distance_only, graph=g, source=i)
        dijkstra = get_runtime(dijkstra_distance_only, graph=n, source=i)
        bmssp = get_runtime(bmssp_distance_only, graph=n, source=i, block_capacity=block_capacity, batch_size=batch_size)

        dijkstrax_res = {k: f"{v:.5f}" for k, v in dijkstrax[0].items()}
        dijkstra_res = {k: f"{v:.5f}" for k, v in dijkstra[0].items()}
        bmssp_res = {k: f"{v:.5f}" for k, v in bmssp[0].items()}


        assert dijkstrax_res == dijkstra_res

        for k, v in bmssp_res.items():
            if dijkstrax_res[k] != v:
                print(dijkstrax_res[k], v)


        assert dijkstrax_res == bmssp_res

    break




    



print(results)