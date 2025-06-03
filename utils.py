
import math

def dtype_to_byte(dtype):
    if dtype in ["fp32"]:
        return 4
    elif dtype in ["fp16", "bfp16"]:
        return 2
    elif dtype in ["fp8", "int8"]:
        return 1
    elif dtype in ["fp4", "int4"]:
        return 0.5
    else:
        raise NotImplementedError

def byte_to_str(byte):
    if byte >= 1024*1024*1024:
        return "{:.2f} GB".format(byte/(1024*1024*1024))
    elif byte >= 1024*1024:
        return "{:.2f} MB".format(byte/(1024*1024))
    elif byte >= 1024:
        return "{:.2f} kB".format(byte/1024)
    else:
        return "{:.2f} B".format(byte)

def mac_to_str(macs):
    if macs >= 1024*1024*1024:
        return "{:.2f} GMAC".format(macs/(1024*1024*1024))
    elif macs >= 1024*1024:
        return "{:.2f} MMAC".format(macs/(1024*1024))
    elif macs >= 1024:
        return "{:.2f} kMAC".format(macs/1024)
    else:
        return "{:.2f} MAC".format(macs)

def intceil(val):
    return int(math.ceil(val))

def colored_text(txt, color=None):
    return  '\033[92m' + str(txt) + '\033[0m'