#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash agent_scripts/patch_assembler_flow_grad.sh
bash agent_scripts/run_assembler_core_debug.sh
