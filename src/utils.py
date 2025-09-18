import zipfile
import random
import numpy as np
import networkx as nx
from scipy.stats import entropy


LND_RISK_FACTOR = 0.000000015
LOG_SPACE = np.logspace(0, 7, 10**6)


def write_zip(file_name, zip_file_path=None):
    if zip_file_path is None:
        zip_file_path = file_name + '.zip'
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as myzip:
        myzip.write(file_name)
    

def read_zip(zip_file_path, 
             use_file_name=None, 
             use_file_index=0, 
             print_file_names=False):
    fname = None
    zf = zipfile.ZipFile(zip_file_path)
    if use_file_name is not None:
        for f in zf.infolist():
            f = f.filename
            if print_file_names:
                print(f)
            if use_file_name in f:
                fname = f
                break
    else:
        fname = zf.infolist()[use_file_index].filename
        if print_file_names:
            print(fname)
    if fname is not None:
        return zf.open(fname)


def degrees(g):
    return np.array(g.degree)[:, 1].astype('float64')


def distribution_entropy(dist, base=2):
    values, counts = np.unique(dist, return_counts=True)
    pk = counts / counts.sum()
    h = entropy(pk, base=base)
    k = len(values)
    h_norm = h / np.log(k) * np.log(base)
    return h, h_norm


def weight_function(g, u, v, amount=508484): # LND

    def calc_bias(w):
        # we don't consider realistic failure heuristic at this point
        w += 10 ** -9
        return w + 0.0005 * max(1.0, w) * random.random()

    fee = g.edges[u, v]['fee_base_sat'] + amount * g.edges[u, v]['fee_rate_sat']
    weight = (amount + fee) * g.edges[u, v]['delay'] * LND_RISK_FACTOR + fee
    weight += calc_bias(weight)
                                                                                      
    assert weight > 0.
    
    return weight


def random_amount(): # SAT
        # Возвращает массив значений от 10^0 = 1 до 10^7, равномерно распределенных на логарифмической шкале
        # https://coingate.com/blog/post/lightning-network-bitcoin-stats-progress
        # The highest transaction processed is 0.03967739 BTC, while the lowest is 0.000001 BTC. The average payment size is 0.00508484 BTC;
        # highest: 3967739.0 SAT
        # average: 508484.0 SAT
        # lowest: 100.0 SAT
        return LOG_SPACE[random.randrange(0, 10**6)]


def graph_to_dict(g):
    return dict(nodes=[(n, g.nodes[n]) for n in g.nodes],
                edges=[(u, v, g.edges[u, v]) for u, v in g.edges])


def graph_from_dict(d):
    g = nx.Graph()
    g.add_nodes_from(d['nodes'])
    g.add_edges_from(d['edges'])
    return g
