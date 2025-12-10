
import numpy as np
import torch 
import matplotlib.pyplot as plt 
import logging 
import json 

class MoEGateModel:
    def __init__(self, num_experts_per_tok, n_routed_experts, layer_ids, workload_model) -> None:
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.layer_ids = [self.strip_layerid(layer_id) for layer_id in layer_ids]
        self.workload_model = workload_model

        if workload_model == "uniform":
            self.probs = {}
            for l, layer_id in enumerate(self.layer_ids):
                self.probs[layer_id] = np.ones(shape=[n_routed_experts])/n_routed_experts

        elif workload_model == "identical":
            pass

        elif workload_model == "empirical_mmlu":
            with open("bincounts.json", "r") as f:
                bincounts = json.load(f)

            self.probs = {}
            for l, layer_id in enumerate(self.layer_ids):
                self.probs[layer_id] = bincounts[str(l)] / np.sum(bincounts[str(l)])
        else:
            raise NotImplementedError

        self.iter_id = None
        self.expert_routings = {}
        
    def new_iter(self, iter_id, bsz, seqlen):
        self.global_bsz = bsz
        self.iter_id = iter_id

        if self.iter_id in self.expert_routings:
            return 
        
        logging.info("MoEGateModel new iter: {} with bsz: {} and seqlen: {}".format(iter_id, bsz, seqlen))

        if self.workload_model == "identical":
            num_tokens = bsz * seqlen
            assert self.num_experts_per_tok*num_tokens % self.n_routed_experts == 0, "num_experts_per_tok * bsz * seqlen must be divisible by n_routed_experts"
            repeat_factor = self.num_experts_per_tok * num_tokens // self.n_routed_experts
            
            # when bincount is calculated with np.count_nonzero(expert_routings == expert_id),
            # effective batch size is going to be equal to repeat_factor 
            routings = np.repeat(np.arange(0,self.n_routed_experts), repeat_factor)
            
            self.expert_routings[self.iter_id] = {}
            for layer_id in self.layer_ids:
                np.random.shuffle(routings)
                self.expert_routings[self.iter_id][layer_id] = routings.reshape([self.num_experts_per_tok, bsz*seqlen])
        elif self.workload_model in ["empirical_mmlu", "uniform"]:
            self.expert_routings[self.iter_id] = {}
            for layer_id in self.layer_ids:
                self.expert_routings[self.iter_id][layer_id] = np.zeros(shape=[self.num_experts_per_tok, bsz*seqlen], dtype=np.int32)
                for i in range(bsz*seqlen):
                    self.expert_routings[self.iter_id][layer_id][:, i] = np.random.choice(a=np.arange(0,self.n_routed_experts), size=[self.num_experts_per_tok], replace=False, p=self.probs[layer_id])
        else:
            raise NotImplementedError
        
    def strip_layerid(self, layer_id):
        # Strip the prefix (e.g., "rank0_") from the layer ID
        return "_".join(layer_id.split("_")[1:])

    def get_expert_routings(self, layer_id):
        return self.expert_routings[self.iter_id][self.strip_layerid(layer_id)]

    def get_bincounts(self, layer_id, expert_id):
        expert_routings = self.get_expert_routings(layer_id)
        return np.count_nonzero(expert_routings == expert_id)

    def get_mapping_by_batchids(self, layer_id, batch_ids):
        expert_routings = self.get_expert_routings(layer_id)
        return expert_routings[:, batch_ids]

moe_gate_model = None

def get_moe_gate_model(num_experts_per_tok = None, n_routed_experts = None, layer_ids = None, workload_model = None):
    global moe_gate_model
    if moe_gate_model is None:
        moe_gate_model = MoEGateModel(num_experts_per_tok, n_routed_experts, layer_ids, workload_model)
    return moe_gate_model

if __name__=="__main__":
    layer = "layer0"
    moe_gate = MoEGateModel(8, 256, [layer], workload_model="uniform")
    moe_gate.new_iter(0, 16, 64)
    bincounts = [moe_gate.get_bincounts(layer, e) for e in range(256)]

    for i in range(100):
        print(sum(bincounts))

    exit()

    for workload_model in ["uniform", "empirical_mmlu"]:
        plt.figure()

        layer = "layer0"

        moe_gate = MoEGateModel(8, 256, [layer], workload_model)
        moe_gate.new_iter(0, 32, 1000)

        bincounts = [moe_gate.get_bincounts(layer, e) for e in range(256)]
        
        sorted_bins = np.sort(bincounts)

        plt.bar(range(len(sorted_bins)), sorted_bins[::-1])
        plt.savefig("out/moe_workload_{}.png".format(workload_model))
