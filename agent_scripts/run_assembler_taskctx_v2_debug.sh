#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash agent_scripts/patch_assembler_context_softflow.sh
bash agent_scripts/patch_multitask_shared_prototypes.sh
bash agent_scripts/patch_audio_head_context_tokens.sh
bash agent_scripts/patch_task_context_v2.sh
bash agent_scripts/run_assembler_multitask_headctx_debug.sh
