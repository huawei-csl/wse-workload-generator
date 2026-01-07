
from typing import Optional, List

class InstructionSet:
    _all_instructions = [
        "READ",
        "WRITE",
        "GEMM",
        "ADD"
    ]

    '''
    Get all supported instruction types.
    '''
    @classmethod
    def get_all_instructions(cls) -> List[str]:
        return cls._all_instructions
    

    '''
    READ instruction to read data from a memory bank.
    Args:
        bank_id: Memory bank ID to read from
        size: Size of data to read in bytes
        comment: Optional comment for the instruction
    Returns:
        Formatted READ instruction string
    '''
    @classmethod
    def READ(cls, bank_id: int, size: int, comment: Optional[str] = None) -> str:
        if comment is None:
            comment = ""
        return "READ {} {}\t\t;{}".format(bank_id, size, comment)\
    
    '''
    WRITE instruction to write data to a memory bank.
    Args:
        bank_id: Memory bank ID to write to
        size: Size of data to write in bytes
        comment: Optional comment for the instruction
    Returns:
        Formatted WRITE instruction string
    '''
    @classmethod
    def WRITE(cls, bank_id: int, size: int, comment: Optional[str] = None) -> str:
        if comment is None:
            comment = ""
        return "WRITE {} {}\t\t;{}".format(bank_id, size, comment)
    
    '''
    GEMM instruction to perform matrix multiplication on matrices of given dimensions.
    Args:
        dims: List of dimensions [M, K, N] for the GEMM operation
        comment: Optional comment for the instruction
    Returns:
        Formatted GEMM instruction string
    '''
    @classmethod
    def GEMM(cls, dims: List[int], comment: Optional[str] = None) -> str:
        assert len(dims) == 3, "GEMM instruction requires 3 dimensions."
        if comment is None:
            comment = ""
        return "GEMM {} {} {}\t\t;{}".format(dims[0], dims[1], dims[2], comment)

    '''
    ️ADD instruction to perform element-wise addition on vectors/matrices of given dimensions.
    ️Args:
        dims: List of dimensions for the addition operation
        comment: Optional comment for the instruction
    ️Returns:
        Formatted ADD instruction string
    '''
    @classmethod
    def ADD(cls, dims: List[int], comment: Optional[str] = None) -> str:
        assert len(dims) >= 1, "ADD instruction requires at least 1 dimension."
        if comment is None:
            comment = ""
        return "ADD {}\t\t;{}".format(" ".join(str(d) for d in dims), comment) 
    
    '''
    DMA transfer instruction to copy data from source memory bank to destination memory bank.
    Args:
        src_bank_id: Source memory bank ID
        dst_bank_id: Destination memory bank ID
        size: Size of data to copy in bytes
        comment: Optional comment for the instruction
    Returns:
        Formatted COPY instruction string
    '''
    @classmethod
    def COPY(cls, src_bank_id: int, dst_bank_id: int, size: int, comment: Optional[str] = None) -> str:
        if comment is None:
            comment = ""
        return "COPY {} {} {}\t\t;{}".format(src_bank_id, dst_bank_id, size, comment)
    
    '''
    Parse an instruction string into its components.
    ️Args:
    ️    instruction_str: Instruction string to parse
    ️Returns:
        Tuple representing the instruction components
    '''
    @classmethod
    def parse(cls, instruction_str: str):
        comment = instruction_str.split(";")[1]
        parts = instruction_str.split(";")[0].strip().split()
        instr_type = parts[0]
        if instr_type == "READ":
            bank_id = int(parts[1])
            size = int(parts[2])
            return ("READ", bank_id, size, comment)
        elif instr_type == "WRITE":
            bank_id = int(parts[1])
            size = int(parts[2])
            return ("WRITE", bank_id, size, comment)
        elif instr_type == "GEMM":
            M = int(parts[1])
            K = int(parts[2])
            N = int(parts[3])
            return ("GEMM", M, K, N, comment)
        elif instr_type == "ADD":
            dims = list(map(int, parts[1:]))
            return ("ADD", dims, comment)
        elif instr_type == "COPY":
            src_bank_id = int(parts[1])
            dst_bank_id = int(parts[2])
            size = int(parts[3])
            return ("COPY", src_bank_id, dst_bank_id, size, comment)
        else:
            raise ValueError("Unknown instruction type: {}".format(instr_type)) 
