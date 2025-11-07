
import os 
import csv

class ComputeGraph:
    def __init__(self) -> None:
        self.nodes = {}

    def add_node(self, op, inputs, outputs, attrs):
        op_id = op.uid
        assert op_id not in self.nodes, "Operation ID {} already exists in the compute graph".format(op_id)
        self.nodes[op_id] = {
            "op_type": op.__class__.__name__,
            "inputs": [t.uid for t in inputs],
            "outputs": [t.uid for t in outputs],
            "attrs": attrs
        }

    def dump(self, fname):
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
            
        with open(fname, "w") as f:
            fieldnames = ["uid", "op_type", "inputs", "outputs", "attrs"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writerows([{"uid":"uid", "op_type":"Op Type", "inputs":"Inputs", "outputs":"Outputs", "attrs":"Attributes"}])
            writer.writerows([{"uid": uid} | self.nodes[uid] for uid in self.nodes])


compute_graph = None

def get_compute_graph():
    global compute_graph
    if compute_graph is None:
        compute_graph = ComputeGraph()
    return compute_graph

def reset_compute_graph():
    global compute_graph
    compute_graph = None