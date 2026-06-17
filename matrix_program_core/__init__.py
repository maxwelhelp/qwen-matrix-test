"""Universal transferable matrix-program core.

This package is the v4 direction: pretrain the same core weights that are later
used with task-specific input adapters and heads.
"""

from .universal_core import CoreAux, ProgramCoreConfig, UniversalMatrixProgramCore

__all__ = ["CoreAux", "ProgramCoreConfig", "UniversalMatrixProgramCore"]
