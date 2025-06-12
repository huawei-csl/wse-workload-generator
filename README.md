
Simulates LLM inference and generates computational requirements for each operation per node in terms of   
1) FLOPS
2) HBM read bytes
3) Interconnect send + receive 

Supported parallelism strategies:
1) Data parallel (DP)
2) Tensor parallel (TP)
3) Pipeline parallel (PP)
4) Sequence parallel (SP)
5) Expert parallel (EP)

It supports different parallelization strategies for Attention and FFN blocks. For example, **#nodes**: 256, **ATTN:** DP=32,SP=2,TP=8; **FFN:** DP=1,TP=1,EP=256

Modelled layers:
1) Linear
2) Self-attention (GQA, MLA-naive, MLA-absorb)
3) Allreduce
4) Alltoall

Supported architectures:
1) Llama GQA
2) DeepSeekv3

To run the simulation:
1) Make sure the config.json (downloaded from HF) for the target model is stored under configs/
2) Enter system properties in system.json
3) Run main.py with the desired arguments. For example:   
```
python main.py --model_config configs/deepseekv3.json --system_config configs/system.json --bsz 1024 --prefill_len 2048 --decode_len 10 --only_decode 1 --simplified_decode 1 --dtype fp16
```

The simulator will generate various system requirements (i.e., **# of flops, memory reads, network requirements**) for each node and for each inference step (prefill + decode) as csv files under **out/**

The logs are written to **./out.log**

For argument descriptions, run:
```
python main.py --help
```


Run sanity checks, which perform decode for various parallelization strategies and check if the results are expected or not:
```
py.test
```



TODOs: 
