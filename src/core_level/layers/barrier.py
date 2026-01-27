
import logging 
import hashlib

from typing import List
from src.core_level.common.wafer import Core
from src.core_level.common.isa import InstructionSet

class Barrier:
    def __init__(self, id, group) -> None:
        self.id = id
        self.group = group # list of node ids participating in the barrier
        self.mapped_core = None
        logging.debug("Barrier {} is created with a group {}.".format(self.id, self.group))

    def map_to_core(self, core: Core):
        assert self.mapped_core is None, "Barrier {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("Barrier {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))
    
    
    def get_traces(self) -> List[str]:
        traces = []

        hash = hashlib.md5((self.id + "_".join(map(str, self.group))).encode()).hexdigest()[:4]
        traces.append(InstructionSet.BARRIER(hash, self.id))
        return traces

