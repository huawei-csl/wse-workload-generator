

class Generator:
    def __init__(self, models) -> None:
        self.models = models

    def generate(self, bsz, prefill_len, decode_len, only_decode, simplified_decode):
        ## PREFILL
        if not only_decode:
            for model in self.models:
                model.forward(bsz, seqlen=prefill_len, ctx_len=0, iter_id=0)

        ## DECODE
        if simplified_decode:
            iter_ids = [0, decode_len-1] # only first and last decode iteration, rest can be interpolated
        else:
            iter_ids = list(range(0, decode_len)) # full run

        for i in iter_ids:
            for model in self.models:
                ctx_len = prefill_len + i
                model.forward(bsz, seqlen=1, ctx_len=ctx_len, iter_id=i+1)

