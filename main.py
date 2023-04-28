
# pipem nainstalovat 
# https://github.com/JakubSido/adthelpers
# pip install git+

import geopy.distance
import adthelpers
import networkx as nx
from queue import PriorityQueue
import numpy as np
import json
import random
np.random.seed(19680801)

def generate_graph():
    graph = nx.grid_2d_graph(3, 3)

    for e in graph.edges:
        s, d = e
        graph[s][d]['weight'] = random.randint(0, 100)
    print(nx.node_link_data(graph))
    return graph


def load_graph(filepath):
    G = nx.DiGraph()
    # implement
    # bude priste
    return G


def print_queue(q : PriorityQueue):
    pass
    
def main():
    
    graph = generate_graph()
    spanning_tree = set()
    closed = set()
    q = PriorityQueue()

    distances = dict()
    predecessors = dict()

    for n in graph.nodes:
        distances[n] = float("inf")
        predecessors[n] = None


    q.put((0, (0,0))) # vzdalenost, konkretni node

    distances[(0,0)] = 0

    painter = adthelpers.painter.Painter(graph, q, closed, None, distances= distances)

    while(q.qsize() > 0):

        current_distance, active_node = q.get()
        print(graph.edges(active_node))
        if active_node in closed: # this-ucitel
            print("Konec prochazeni grafu")
            continue

        edges = graph.edges(active_node)

        for e in edges:
            soused = e[1]
            print(soused, distances[soused])
            weight = graph.get_edge_data(active_node, soused)['weight'] # vaha cesty od aktualniho nodu k sousedovi

            new_distance = current_distance + weight

            if(new_distance < distances[soused]):
                # print(current_distance, "+", weight , "<" , distances[soused])
                distances[soused] = new_distance
                predecessors[soused] = active_node
                # print(soused)
                q.put((new_distance, soused))
                # print(distances[soused])


            # sousedi_souseda = graph.edges(e[1])
            # print(sousedi_souseda)
            # fr, to = e[0], e[1] # To jsou dva body
            # w = graph.get_edge_data(fr, to)['weight']
            # q.put((w, to, e))

        closed.add(active_node)
    
    print(distances)
    print(predecessors)
    get_path((2,2), predecessors)
    painter.draw_graph(active_node)

def get_path(cil, predecessors):
    path = predecessors[cil]
    cesta = list()

    while(path is not None):
        cesta.append(path)
        path = predecessors[path]

    print("Celkova cesta: ",cesta)
        
def load_graph():
    pass
# G = nx.DiGraph()
# G.add_edge(source, dest, weight = 4)

# aproxiamce kouli,  vzdalenost dopocitat z line stringu, knihovna 
# 4651 hradek
# 4659 zbuch - vypocitat jejich cestu
# v edges je automatickz jednosmerna cesta
# sirka a delka jsou prohozeni v souborech


if __name__ == '__main__':


    coords_1 = (52.2296756, 21.0122287)
    coords_2 = (52.406374, 16.9251681)

    print(geopy.distance.geodesic(coords_1, coords_2).km)

    main()
    # hledani minimalni kostry grafu
    # todo - networkX knihovna
