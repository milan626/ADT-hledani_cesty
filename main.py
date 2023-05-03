
# pipem nainstalovat 
# https://github.com/JakubSido/adthelpers
# pip install git+

from shapely.wkt import loads
import geopy.distance as geoD
import adthelpers
import networkx as nx
from queue import PriorityQueue
import numpy as np
import json
import random

from tqdm import tqdm
np.random.seed(19680801)


class Node:
    def __init__(self, linestring) -> None:
        self.linestring = linestring

    def __repr__(self) -> str:
        return str(self.get_cordinates())
    
    def __str__(self) -> str:
        return str(self.get_cordinates())

    def get_cordinates(self):
        temp  = self.linestring.replace('POINT(', "")
        temp = temp.replace(')', "")
        y, x = temp.split(" ")
        return (float(x), float(y))

def generate_graph():
    graph = nx.grid_2d_graph(3, 3)

    for e in graph.edges:
        s, d = e
        graph[s][d]['weight'] = random.randint(0, 100)
    print(nx.node_link_data(graph))
    return graph


def load_graphs(filepath):
    G = nx.DiGraph()
    # implement
    # bude priste
    return G


def print_queue(q : PriorityQueue):
    pass
    
def dijkstra(start_node, end_node, graph):
    
    # spanning_tree = set()
    closed = set()
    q = PriorityQueue()

    distances = dict()
    predecessors = dict()

    for n in graph.nodes:
        distances[n] = float("inf")
        predecessors[n] = None


    q.put((0, start_node)) # vzdalenost, konkretni node

    distances[start_node] = 0

    # painter = adthelpers.painter.Painter(graph, q, closed, None, distances= distances)

    while(q.qsize() > 0):

        current_distance, active_node = q.get()
        # print(graph.edges(active_node))
        if active_node in closed: # this-ucitel
            # print("Konec prochazeni grafu")
            continue

        edges = graph.edges(active_node)

        for e in edges:
            soused = e[1]
            # print(soused, distances[soused])
            weight = graph.get_edge_data(active_node, soused)['weight'] # vaha cesty od aktualniho nodu k sousedovi

            new_distance = current_distance + weight

            if(new_distance < distances[soused]):
                # print(current_distance, "+", weight , "<" , distances[soused])
                distances[soused] = new_distance
                predecessors[soused] = active_node
                # print(soused)
                q.put((new_distance, soused))
                # print(distances[soused])
        # painter.draw_graph(active_node)


            # sousedi_souseda = graph.edges(e[1])
            # print(sousedi_souseda)
            # fr, to = e[0], e[1] # To jsou dva body
            # w = graph.get_edge_data(fr, to)['weight']
            # q.put((w, to, e))

        closed.add(active_node)
    
    # print(distances)
    # print(predecessors)
    return distances, predecessors

def get_path(cil, predecessors):
    path = predecessors[cil]
    cesta = list()

    while(path is not None):
        cesta.append(path)
        path = predecessors[path]

    print("Celkova cesta: ",cesta)
    return cesta
        
def load_graph(edges_path, nodes_path):
    G = nx.DiGraph()
    nodes = load_nodes(nodes_path)

    with open(edges_path, "r+") as file:
        file.readline()
        lines = file.readlines()

        for line in tqdm(lines):
            if line.strip() == "":
                continue

            edge_id,source,target,capacity,isvalid,WKT = line.split(",", 5)
            G.add_edge(int(source), int(target), weight=get_distance(WKT), edge_data=(edge_id, WKT))

    return G, nodes



# G = nx.DiGraph()
# G.add_edge(source, dest, weight = 4)

# aproxiamce kouli,  vzdalenost dopocitat z line stringu, knihovna 
# 4651 hradek
# 4559 zbuch - vypocitat jejich cestu
# v edges je automatickz jednosmerna cesta
# sirka a delka jsou prohozeni v souborech

def load_nodes(filepath):
    nodes = dict()
    with open(filepath, "r+") as file:
        file.readline()
        lines = file.readlines()
        for line in lines:
            node_id, linestring = line.split(",")
            linestring = linestring.replace('\"', "").strip()
            nodes[int(node_id)] = Node(linestring)

    return nodes



def get_distance(long_ls):
    # a number of other elipsoids are supported
    distance = 0
    geoD.geodesic.ELLIPSOID = 'WGS-84'
    d = geoD.distance

    list_of_ls = split_LineString(long_ls)

    for ls in list_of_ls:
        line = loads(ls)
        # convert the coordinates to xy array elements, compute the distance
        dist = d(line.xy[0], line.xy[1])
        # print(dist.meters)
        distance += dist.meters

    return distance

def split_LineString(original):
    result = list()
    prefix = "LINESTRING("
    postfix = ")"
    original = original.replace(prefix, "").strip()
    original = original.replace(postfix, "")
    original = original.replace('\"', '')
    splitted = original.split(",")

    for i in range(0, len(splitted)-1):
        long1, lat1 = splitted[i].split(" ")
        long2, lat2 = splitted[i+1].split(" ")
        newLS = prefix + lat1 + " " + lat2 + "," + long1 + " " + long2 + postfix
        # newLS = prefix + lat1 + " " + long1 + "," + lat2 + " " + long2 + postfix
        # print(newLS)
        result.append(newLS)

    return result

def get_final_distance(path, nodes, graph):
    final_distance = 0
    i = 0

    with open("output.txt", "w") as file:
        with open("output_edges.txt", "w") as file_e:
            for p in range(len(path)-1):
                # print(path[i])
                to_node_id = int(path[i])
                i += 1
                # print(path[i])
                from_node_id = int(path[i])
                edge_data = graph.get_edge_data(from_node_id, to_node_id)
                length = edge_data['weight']
                edge_id, WKT = edge_data['edge_data']
                WKT = WKT.replace('\"', "")

                file_e.write(WKT + ",")

                text = nodes[to_node_id].linestring + ",\n"
                # text = nodes[from_node_id].linestring + ", "+ nodes[to_node_id].linestring + ",\n"
                file.write(text)

                final_distance += length

            file.write(nodes[path[-1]].linestring)


    print("Celkova vzdalenost: " + str(final_distance))    
    print(len(path), i)


    # linestring = "LINESTRING(13.2493302001435 49.764380533239,13.249074682493 49.7658087519211,13.2488225160078 49.7674230526761,13.2485899684292 49.7692554591859,13.248584957672 49.7692712940434,13.2483006280357 49.7698227628131,13.2480651174741 49.7705679279674,13.2479923746268 49.7715163332597,13.2482168296308 49.7721304869104,13.2486411097935 49.7726666339142,13.2486415190555 49.7726671566612,13.2494821842308 49.7737525730256,13.2500302202536 49.7741557317799)"
    # print(get_distance(linestring))
def main():
    graph, nodes = load_graph("data/pilsen_edges.csv", "data/pilsen_nodes.csv")
    start_node = 4559
    end_node = 4651

    distances, predecessors = dijkstra(start_node, end_node, graph)
    path = get_path(end_node, predecessors)
    
    get_final_distance(path, nodes, graph)
    

if __name__ == '__main__':
    main()

    # print("Data loaded")
    # while(True):
    #     inp = input()
    #     edges = graph.edges(int(inp))
    #     print(edges)

    #     print("Nodes:")
    #     for start,end in edges:
    #         print("-------")
    #         print(nodes[int(start)].linestring)
    #         print(nodes[int(end)].linestring)
    #         print(graph.get_edge_data(start, end))
    # print("(////////////////////////////")

    # graph = generate_graph()
    # main((0,0), (2,2), graph)

    """

    """