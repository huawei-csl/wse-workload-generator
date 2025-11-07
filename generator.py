

from tensor import Tensor
from compute_graph import reset_compute_graph

class Generator:
    def __init__(self) -> None:
        pass

    def prefill(self, models, bsz, prefill_len):
        reset_compute_graph()

        ## PREFILL
        for model in models:
            queries = Tensor("queries", [bsz, prefill_len, model.hidden_size])
            model.forward(queries, ctx_len=0, iter_id=0)

    def decode(self, models, bsz, prefill_len, decode_len, simplified_decode):
        ## DECODE
        if simplified_decode:
            iter_ids = [0, decode_len-1] # only first and last decode iteration, rest can be interpolated
        else:
            iter_ids = list(range(0, decode_len)) # full run

        for i in iter_ids:
            for model in models:
                reset_compute_graph()
                ctx_len = prefill_len + i
                seqlen_q = 1

                queries = Tensor("queries", [bsz, seqlen_q, model.hidden_size])
                model.forward(queries, ctx_len=ctx_len, iter_id=i+1)

