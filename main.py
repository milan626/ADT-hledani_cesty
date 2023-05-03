
# pipem nainstalovat 
# https://github.com/JakubSido/adthelpers
# pip install git+

from wktplot.plots.osm import OpenStreetMapsPlot

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
    
def dijkstra(start_node, graph):
    closed = set()
    q = PriorityQueue()

    distances = dict()
    predecessors = dict()

    for n in graph.nodes:
        distances[n] = float("inf")
        predecessors[n] = None


    q.put((0, start_node)) # vzdalenost, konkretni node

    distances[start_node] = 0

    while(q.qsize() > 0):

        current_distance, active_node = q.get()
        if active_node in closed:
            continue

        edges = graph.edges(active_node)

        for e in edges:
            soused = e[1]
            weight = graph.get_edge_data(active_node, soused)['weight'] # vaha cesty od aktualniho nodu k sousedovi

            new_distance = current_distance + weight

            if(new_distance < distances[soused]):
                distances[soused] = new_distance
                predecessors[soused] = active_node
                q.put((new_distance, soused))


        closed.add(active_node)

    return distances, predecessors

def get_path(cil, predecessors):
    path = predecessors[cil]
    cesta = list()

    while(path is not None):
        cesta.append(path)
        path = predecessors[path]

    # print("Celkova cesta: ",cesta)
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
    distance = 0
    geoD.geodesic.ELLIPSOID = 'WGS-84'
    d = geoD.distance

    list_of_ls = split_LineString(long_ls)

    for ls in list_of_ls:
        line = loads(ls)
        dist = d(line.xy[0], line.xy[1])
        distance += dist.meters

    return distance

def invert_linestring(original):
    prefix = "LINESTRING("
    postfix = ")"
    result = prefix
    original = original.replace(prefix, "").strip()
    original = original.replace(postfix, "")
    original = original.replace('\"', '')    
    splitted = original.split(",")

    for i in range(0, len(splitted)):
        long, lat = splitted[i].split(" ")
        result += lat + " " + long

        if(i < len(splitted)-1):
            result += ", "

    result += postfix
    return result    


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
        result.append(newLS)

    return result

def get_final_distance(path, nodes, graph, plot):
    final_distance = 0
    i = 0

    with open("output.txt", "w") as file:
        with open("output_edges.txt", "w") as file_e:
            for p in range(len(path)-1):
                to_node_id = int(path[i])
                i += 1
                from_node_id = int(path[i])
                edge_data = graph.get_edge_data(from_node_id, to_node_id)
                length = edge_data['weight']
                edge_id, WKT = edge_data['edge_data']

                plot.add_shape(invert_linestring(WKT), line_color="blue", line_alpha=0.8, line_width=5)

                file_e.write(WKT + ",")

                text = nodes[to_node_id].linestring + ",\n"
                file.write(text)

                final_distance += length

            file.write(nodes[path[-1]].linestring)

    plot.save()
    return final_distance

def main():
    graph, nodes = load_graph("data/pilsen_edges.csv", "data/pilsen_nodes.csv")
    start_node = 4569
    end_node = 4651

    distances, predecessors = dijkstra(start_node, graph)

    path = get_path(end_node, predecessors)
    
    plot = OpenStreetMapsPlot("Open Street Map Plot",height=1000,width=1200)

    # Zároveň vytvoří 3 soubory, 
    final_dist = get_final_distance(path, nodes, graph, plot)
    print("Celkova vzdalenost: " + str(final_dist) + "km")
    plot.show()


    while(True):
        # aby se nezavrel prohlizec
        pass
    

    

if __name__ == '__main__':
    main()

    """
    test - hledani nejkratsi cesty a minimalni kostry

    """

    
# G = nx.DiGraph()
# G.add_edge(source, dest, weight = 4)

# aproxiamce kouli,  vzdalenost dopocitat z line stringu, knihovna 
# 4651 hradek
# 4559 zbuch - vypocitat jejich cestu
# v edges je automatickz jednosmerna cesta
# sirka a delka jsou prohozeni v souborech