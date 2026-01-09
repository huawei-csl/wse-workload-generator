
import csv 

graph = None

class Graph:
    def __init__(self, iter, num_nodes, ops=None) -> None:
        if ops is None:
            self.ops = self.load_graph(iter, num_nodes)
        else:
            self.ops = ops
            
    def load_graph(self, iter, num_nodes):
        ops = {}
        
        for node_id in range(num_nodes):
            ops[node_id] = {}

            csv_fname = f"out_graph/node_{node_id}/{iter}.csv"
            with open(csv_fname, mode="r") as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=";")  # Automatically uses the first row as headers
                for row in csv_reader:
                    op_id = row["uid"]
                    op_type = row["Op Type"]
                    inputs = row["Inputs"][1:-1].replace("'","").split(", ")
                    outputs = row["Outputs"][1:-1].replace("'","").split(", ")

                    ops[node_id][op_id] = {"type": op_type, "inputs": inputs, "outputs": outputs}

        return ops

    def get_op(self, node_id, op_id):
        return self.ops[node_id][op_id]

def init_graph(iter, num_nodes):
    global graph
    assert graph is None, "Graph is already initialized"
    graph = Graph(iter, num_nodes)
    return graph

def get_compute_graph():
    global graph
    assert graph is not None, "Graph is not initialized"
    return graph
