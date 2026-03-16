import time 
import logging 

from src.node_level.common.tensor import Tensor
from src.node_level.common.compute_graph import reset_compute_graph

class Generator:
    def __init__(self) -> None:
        pass

    def prefill(self, models, bsz, prefill_len):
        reset_compute_graph()

        ## PREFILL
        for model in models:
            queries = Tensor("queries", model.dist_info.rank, [bsz, prefill_len, model.hidden_size])
            model.forward(queries, ctx_len=0, iter_id=0)

    def decode(self, models, bsz, seqlen_q, prefill_len, decode_len, simplified_decode):
        ## DECODE
        if simplified_decode:
            iter_ids = [0, decode_len-1] # only first and last decode iteration, rest can be interpolated
        else:
            iter_ids = list(range(0, decode_len)) # full run

        for i in iter_ids:
            for model in models:
                start = time.time()

                reset_compute_graph()
                ctx_len = prefill_len + i
                # seqlen_q = 1

                queries = Tensor("queries", model.dist_info.rank, [bsz, seqlen_q, model.hidden_size])
                model.forward(queries, ctx_len=ctx_len, iter_id=i+1)

                logging.info(f"Node {model.dist_info.rank} / {len(models)} iteration {i+1} completed in {time.time() - start:.4f} seconds")