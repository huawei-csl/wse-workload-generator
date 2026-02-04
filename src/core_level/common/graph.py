
import csv 
import networkx as nx

class Node:
    def __init__(self, uid, node_id, op_type, input_tensors, output_tensors, parent) -> None:
        self.uid = uid
        self.node_id = node_id
        self.op_type = op_type
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors

        self.parent = parent
        self.children = []

    def add_child(self, child: "Node"):
        child.add_parent(self)
        self.children.append(child)

    def add_parent(self, parent: "Node"):
        assert self.parent is None, "Parent already set"
        self.parent = parent

    def insert_child(self, child: "Node"):
        for tensor in child.input_tensors:
            if tensor in self.output_tensors:
                self.add_child(child)
        
        for child in self.children:
            child.insert_child(child)

class Edge:
    def __init__(self, srcs: Node, dsts: Node, tensor_uid) -> None:
        self.srcs = srcs
        self.dsts = dsts
        self.tensor_uid = tensor_uid

class Graph:
    def __init__(self, iter, num_nodes, ops=None, dir=None) -> None:
        self.dir = dir
        if ops is None:
            self.ops, self.edges = self.load_ops(iter, num_nodes)
            self.graph = nx.DiGraph()
            self.create_graph()
            # self.draw_graph()
        else:
            self.ops = ops
            self.edges = {}

    def load_ops(self, iter, num_nodes):
        ops = {}
        edges = {}

        for node_id in range(num_nodes):
            ops[node_id] = {}

            csv_fname = f"{self.dir}/node_{node_id}/{iter}.csv"
            with open(csv_fname, mode="r") as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=";")  # Automatically uses the first row as headers
                for row in csv_reader:
                    op_id = row["uid"]
                    op_type = row["Op Type"]
                    input_tensors = row["Inputs"][1:-1].replace("'","").split(", ")
                    output_tensors = row["Outputs"][1:-1].replace("'","").split(", ")

                    ops[node_id][op_id] = {"type": op_type, "inputs": input_tensors, "outputs": output_tensors}

                    for input_tensor in input_tensors:
                        if input_tensor not in edges:
                            edges[input_tensor] = {"srcs": [], "dsts": []}
                        edges[input_tensor]["dsts"].append((node_id, op_id))

                    for output_tensor in output_tensors:
                        if output_tensor not in edges:
                            edges[output_tensor] = {"srcs": [], "dsts": []}
                        edges[output_tensor]["srcs"].append((node_id, op_id))

        return ops, edges

    def create_graph(self):
        root_edges = self.find_root_edges()
        assert root_edges == [f"{node_id}:queries" for node_id in range(len(self.ops))], "Unexpected root edges: {}".format(root_edges)

        self.root = Node("root", node_id=-1, op_type="root", input_tensors=[], output_tensors=root_edges, parent=None) # Assuming 'node_id:queries' is the input tensor to each node
        self.graph.add_node(self.root)

        nodes = {}
        for node_id in self.ops:
            nodes[node_id] = {}
            for op_id in self.ops[node_id]:
                # node = self.ops[node_id][op_id]
                op_type = self.ops[node_id][op_id]["type"]
                input_tensors = self.ops[node_id][op_id]["inputs"]
                output_tensors = self.ops[node_id][op_id]["outputs"]
                node = Node(op_id, node_id, op_type, input_tensors, output_tensors, None)
                self.graph.add_node(node)
                nodes[node_id][op_id] = node

        for tensor in self.edges:
            if len(self.edges[tensor]["srcs"]) == 0:
                for dst_node_id, dst_op_id in self.edges[tensor]["dsts"]:
                    self.graph.add_edge(self.root, nodes[dst_node_id][dst_op_id])

            for src_node_id, src_op_id in self.edges[tensor]["srcs"]:
                for dst_node_id, dst_op_id in self.edges[tensor]["dsts"]:
                    self.graph.add_edge(nodes[src_node_id][src_op_id], nodes[dst_node_id][dst_op_id])

        reachable = nx.descendants(self.graph, self.root) | {self.root}
        missing = set(self.graph.nodes()) - set(reachable)
        assert missing == set(), "Graph contains unreachable nodes."

        assert nx.is_directed_acyclic_graph(self.graph), "Graph contains a cycle"
        
    def find_root_edges(self):
        '''
        Find all edges with no producers, represent input placeholders.
        '''
        root_edges = []
        for edge in self.edges:
            if len(self.edges[edge]["srcs"]) == 0 and "empty" not in edge:
                root_edges.append(edge)
        return root_edges

    def get_op(self, node_id, op_id):
        return self.ops[node_id][op_id]
    
    def get_topological_order(self):
        return list(nx.topological_sort(self.graph))

    def draw_graph(self, filepath="graph.png"):
        import matplotlib.pyplot as plt

        pos = nx.bfs_layout(self.graph, start=self.root)

        plt.figure(figsize=(30, 30)) 
        nx.draw(
            self.graph, 
            pos, 
            with_labels=True, 
            labels={node: node.uid for node in self.graph.nodes()}, 
            node_size=300, 
            node_color="lightblue")
        plt.savefig(filepath)

graph = None

def init_graph(iter, num_nodes, dir):
    global graph
    assert graph is None, "Graph is already initialized"
    graph = Graph(iter, num_nodes, dir=dir)
    return graph

def get_compute_graph():
    global graph
    assert graph is not None, "Graph is not initialized"
    return graph

if __name__=="__main__":
    graph = init_graph("decode0", 4)

