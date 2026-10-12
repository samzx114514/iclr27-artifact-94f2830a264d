from enum import Enum


class ExecutionMode(str, Enum):
    EXACT_REUSE = "exact_reuse"
    FRESH_SOLVE = "fresh_solve"
    REUSE_AND_PATCH = "reuse_and_patch"

