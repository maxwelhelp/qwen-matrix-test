"""Transferable matrix-program cores.

v4 direction: pretrain the same core weights that are later used with
task-specific input adapters and heads.
"""

from .universal_core import CoreAux, ProgramCoreConfig, UniversalMatrixProgramCore
from .assembler_core import AssemblerAux, AssemblerConfig, MatrixProgramAssemblerCore

__all__ = [
    "CoreAux",
    "ProgramCoreConfig",
    "UniversalMatrixProgramCore",
    "AssemblerAux",
    "AssemblerConfig",
    "MatrixProgramAssemblerCore",
]
