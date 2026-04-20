import logging

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Instruction:
    """A single emitted trace instruction with local ids and deps.

    `local_id` is the op-local instruction index starting at 0. The
    Core aggregator rebases it to a file-wide global id on append, so
    each emitting `get_traces()` only needs to number its own output
    contiguously from 0.

    `local_deps` carries the local ids of producer instructions in the
    *same* emitting op whose completion this instruction depends on.
    Cross-op deps are not supported; anything that crossed a barrier
    would be redundant with the barrier itself anyway.
    """
    opcode: str
    args: List[str]
    local_id: int
    local_deps: List[int] = field(default_factory=list)
    comment: str = ""

    def render(self, global_id: int, global_deps: List[int]) -> str:
        parts = [str(global_id), self.opcode, *self.args]
        if global_deps:
            parts.append("[" + " ".join(str(d) for d in global_deps) + "]")
        return " ".join(parts) + "\t\t;{}".format(self.comment)


class InstructionSet:
    _all_instructions = [
        "READ",
        "WRITE",
        "GEMM",
        "ADD",
        "COPY",
        "MULTICAST",
        "BARRIER"
    ]

    @classmethod
    def get_all_instructions(cls) -> List[str]:
        return cls._all_instructions

    @classmethod
    def READ(cls, bank_id: int, size: int, comment: Optional[str], local_id: int, deps: Optional[List[int]] = None) -> Instruction:
        logging.debug("Read {} bytes from bank {}".format(size, bank_id))
        return Instruction(
            opcode="READ",
            args=[str(bank_id), str(size)],
            local_id=local_id,
            local_deps=list(deps) if deps else [],
            comment=comment if comment is not None else "",
        )

    @classmethod
    def WRITE(cls, bank_id: int, size: int, comment: Optional[str], local_id: int, deps: Optional[List[int]] = None) -> Instruction:
        logging.debug("Write {} bytes to bank {}".format(size, bank_id))
        return Instruction(
            opcode="WRITE",
            args=[str(bank_id), str(size)],
            local_id=local_id,
            local_deps=list(deps) if deps else [],
            comment=comment if comment is not None else "",
        )

    @classmethod
    def GEMM(cls, dims: List[int], comment: Optional[str], local_id: int, deps: Optional[List[int]] = None) -> Instruction:
        logging.debug("Perform GEMM with dims {}".format(dims))
        assert len(dims) == 3, "GEMM instruction requires 3 dimensions."
        return Instruction(
            opcode="GEMM",
            args=[str(dims[0]), str(dims[1]), str(dims[2])],
            local_id=local_id,
            local_deps=list(deps) if deps else [],
            comment=comment if comment is not None else "",
        )

    @classmethod
    def ADD(cls, dims: List[int], comment: Optional[str], local_id: int, deps: Optional[List[int]] = None) -> Instruction:
        logging.debug("Perform ADD with dims {}".format(dims))
        assert len(dims) >= 1, "ADD instruction requires at least 1 dimension."
        return Instruction(
            opcode="ADD",
            args=[str(d) for d in dims],
            local_id=local_id,
            local_deps=list(deps) if deps else [],
            comment=comment if comment is not None else "",
        )

    @classmethod
    def COPY(cls, src_bank_id: int, dst_bank_id: int, size: int, comment: Optional[str], local_id: int, deps: Optional[List[int]] = None) -> Instruction:
        logging.debug("Copy {} bytes from {} to {}".format(size, src_bank_id, dst_bank_id))
        return Instruction(
            opcode="COPY",
            args=[str(src_bank_id), str(dst_bank_id), str(size)],
            local_id=local_id,
            local_deps=list(deps) if deps else [],
            comment=comment if comment is not None else "",
        )

    @classmethod
    def MULTICAST(cls, src_bank_id: int, dst_bank_ids: List[int], size: int, comment: Optional[str], local_id: int, deps: Optional[List[int]] = None) -> Instruction:
        logging.debug("Multicast {} bytes from {} to {}".format(size, src_bank_id, dst_bank_ids))
        return Instruction(
            opcode="MULTICAST",
            args=[str(src_bank_id), *[str(d) for d in dst_bank_ids], str(size)],
            local_id=local_id,
            local_deps=list(deps) if deps else [],
            comment=comment if comment is not None else "",
        )

    @classmethod
    def BARRIER(cls, hash, comment: Optional[str], local_id: int) -> Instruction:
        logging.debug("Barrier with uid {}".format(hash))
        return Instruction(
            opcode="BARRIER",
            args=[str(hash)],
            local_id=local_id,
            local_deps=[],
            comment=comment if comment is not None else "",
        )

    @classmethod
    def parse(cls, instruction_str: str):
        """Parse a rendered trace line back into a tuple.

        The rendered format is `<id> <OPCODE> <args...> [<deps>]\\t\\t;<comment>`
        where the `[<deps>]` bracketed block is optional. To stay source-
        compatible with existing callers that destructure the returned tuple
        by positional index (opcode, arg1, arg2, ..., comment), this helper
        strips the leading id and the optional deps block and returns the
        same tuple shape as before. The discarded id/deps are available to
        callers that need them via the dataclass or by pre-parsing; no
        current caller in wse-workload uses them.
        """
        comment = instruction_str.split(";", 1)[1]
        body = instruction_str.split(";", 1)[0].strip()
        tokens = body.split()

        # Strip optional trailing [dep ids] block.
        if tokens and tokens[-1].endswith("]"):
            for i in range(len(tokens) - 1, -1, -1):
                if tokens[i].startswith("["):
                    tokens = tokens[:i]
                    break

        # First token is the instruction id; opcode and args follow.
        assert len(tokens) >= 2, "Malformed instruction: {!r}".format(instruction_str)
        parts = tokens[1:]

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
        elif instr_type == "MULTICAST":
            src_bank_id = int(parts[1])
            dst_bank_ids = list(map(int, parts[2:-1]))
            size = int(parts[-1])
            return ("MULTICAST", src_bank_id, dst_bank_ids, size, comment)
        elif instr_type == "BARRIER":
            uid = str(parts[1])
            return ("BARRIER", uid, comment)
        else:
            raise ValueError("Unknown instruction type: {}".format(instr_type))
