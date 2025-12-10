
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

def divide_equal(val, n_div):
    smallest = val // n_div
    remainder = val - n_div * smallest

    parts = []
    for i in range(n_div):
        if i < remainder:
            parts.append(smallest+1)
        else:
            parts.append(smallest)

    return parts

def colored_text(txt, color=None):
    return  '\033[92m' + str(txt) + '\033[0m'

def hash_string(s, num_digits=8):
    import hashlib
    return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest().upper()[:num_digits]


if __name__=="__main__":
    parts = divide_equal(63, 4)
    print(parts)