
import logging

from typing import List
from utils import intceil
from core_level.layers import UnicastLayer




class AllreduceLayer:
    def __init__(self, uid, node_id, comm_group, dims, wafer, prec) -> None:
        self.uid = uid
        self.comm_group = comm_group
        
        vector_dim = eval("*".join([str(d) for d in dims]))
        next_node = comm_group[(comm_group.index(node_id) + 1) % len(comm_group)]

        # reduce stage
        num_rounds = len(comm_group) - 1
        for round in range(num_rounds):
            UnicastLayer(self.uid + f"reduce_{round}", node_id, next_node, [vector_dim//len(comm_group),], wafer, prec)

        # gather stage
        for round in range(num_rounds):
            UnicastLayer(self.uid + f"gather_{round}", node_id, next_node, [vector_dim//len(comm_group),], wafer, prec)
