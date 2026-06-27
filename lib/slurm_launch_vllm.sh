#!/usr/bin/env bash
# Submit a long-running vLLM server job to Slurm and print the Slurm job id.
#
# Inputs are supplied by lib/server.sh through PERF_EVAL_* environment vars.
# The default path launches non-Ray `vllm serve` through Slurm. If Pyxis is
# available it can run in a Slurm container; clusters with their own launch
# wrapper can use PERF_EVAL_SLURM_SERVER_COMMAND for SRT-Slurm bootstrapping.
set -euo pipefail

require() {
  local name=$1
  local value=${!name:-}
  [[ -n "$value" ]] || { echo "missing required env $name" >&2; exit 2; }
}

shell_quote() {
  printf "%q" "$1"
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_container_runtime() {
  local runtime=${PERF_EVAL_SLURM_CONTAINER_RUNTIME:-auto}
  if [[ "$runtime" != "auto" ]]; then
    printf '%s\n' "$runtime"
    return
  fi
  if srun --help 2>&1 | grep -q -- "--container-image"; then
    printf 'pyxis\n'
  else
    printf 'none\n'
  fi
}

require PERF_EVAL_CONTAINER_NAME
require PERF_EVAL_PORT
require PERF_EVAL_MODEL
require PERF_EVAL_ENDPOINT_FILE
require PERF_EVAL_RESULTS_DIR

mkdir -p "$PERF_EVAL_RESULTS_DIR"

script="${PERF_EVAL_RESULTS_DIR}/${PERF_EVAL_CONTAINER_NAME}.sbatch.sh"
log_file="${PERF_EVAL_RESULTS_DIR}/${PERF_EVAL_CONTAINER_NAME}-%j.log"
nodes="${PERF_EVAL_NUM_NODES:-1}"
gpus_per_node="${PERF_EVAL_GPUS_PER_NODE:-${PERF_EVAL_NUM_GPUS:-1}}"
slurm_gpus_per_node="${PERF_EVAL_SLURM_GPUS_PER_NODE:-$gpus_per_node}"
ntasks_per_node="${PERF_EVAL_SLURM_NTASKS_PER_NODE:-1}"
job_name="${PERF_EVAL_SLURM_JOB_NAME:-$PERF_EVAL_CONTAINER_NAME}"
container_runtime="$(resolve_container_runtime)"
request_gpus="${PERF_EVAL_SLURM_REQUEST_GPUS:-1}"
slurm_gres="${PERF_EVAL_SLURM_GRES:-}"

serve_cmd="vllm serve $(shell_quote "$PERF_EVAL_MODEL") --host 0.0.0.0 --port $(shell_quote "$PERF_EVAL_PORT")"
if [[ -n "${PERF_EVAL_SERVE_ARGS:-}" ]]; then
  serve_cmd+=" ${PERF_EVAL_SERVE_ARGS}"
fi

server_command="${PERF_EVAL_SLURM_SERVER_COMMAND:-}"
if [[ -z "$server_command" ]]; then
  srun_args=(--ntasks=1)
  if is_truthy "$request_gpus"; then
    if [[ -n "$slurm_gres" ]]; then
      srun_args+=(--gres "$slurm_gres")
    elif [[ -n "$slurm_gpus_per_node" ]]; then
      srun_args+=(--gpus-per-node "$slurm_gpus_per_node")
    fi
  fi
  case "$container_runtime" in
    pyxis|container)
      require PERF_EVAL_IMAGE
      srun_args+=(--container-image "$PERF_EVAL_IMAGE")
      if [[ -n "${PERF_EVAL_SLURM_CONTAINER_MOUNTS:-}" ]]; then
        srun_args+=(--container-mounts "$PERF_EVAL_SLURM_CONTAINER_MOUNTS")
      fi
      if [[ "${PERF_EVAL_SLURM_NO_CONTAINER_REMAP_ROOT:-1}" == "1" ]]; then
        srun_args+=(--no-container-remap-root)
      fi
      ;;
    none|native)
      ;;
    *)
      echo "unsupported PERF_EVAL_SLURM_CONTAINER_RUNTIME=$container_runtime" >&2
      exit 2
      ;;
  esac
  if [[ -n "${PERF_EVAL_SLURM_MPI:-pmix}" ]]; then
    srun_args+=(--mpi "${PERF_EVAL_SLURM_MPI:-pmix}")
  fi
  if [[ -n "${PERF_EVAL_SLURM_EXTRA_SRUN_ARGS:-}" ]]; then
    # shellcheck disable=SC2206  # Buildkite env intentionally supplies words.
    extra_srun_args=($PERF_EVAL_SLURM_EXTRA_SRUN_ARGS)
    srun_args+=("${extra_srun_args[@]}")
  fi

  server_command="srun"
  for arg in "${srun_args[@]}"; do
    server_command+=" $(shell_quote "$arg")"
  done
  server_command+=" bash -lc $(shell_quote "$serve_cmd")"
fi

cat > "$script" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail

env_file=$(shell_quote "${PERF_EVAL_ENV_FILE:-}")
if [[ -n "\$env_file" && -f "\$env_file" ]]; then
  while IFS= read -r kv; do
    [[ -z "\$kv" ]] && continue
    export "\$kv"
  done < "\$env_file"
fi

host=\$(hostname -f 2>/dev/null || hostname)
log_path=$(shell_quote "$log_file")
if [[ -n "\${SLURM_JOB_ID:-}" ]]; then
  log_path="\${log_path//%j/\$SLURM_JOB_ID}"
fi
tmp="$(shell_quote "$PERF_EVAL_ENDPOINT_FILE").tmp"
{
  printf 'VLLM_SERVER_HOST=%q\\n' "\$host"
  printf 'VLLM_SERVER_PORT=%q\\n' "$(shell_quote "$PERF_EVAL_PORT")"
  printf 'VLLM_BASE_URL=%q\\n' "http://\${host}:$(shell_quote "$PERF_EVAL_PORT")"
  printf 'VLLM_SLURM_LOG_FILE=%q\\n' "\$log_path"
} > "\$tmp"
mv "\$tmp" "$(shell_quote "$PERF_EVAL_ENDPOINT_FILE")"

exec $server_command
SCRIPT
chmod +x "$script"

sbatch_args=(--parsable --job-name "$job_name" --nodes "$nodes" \
  --ntasks-per-node "$ntasks_per_node" --output "$log_file" --error "$log_file")
if is_truthy "$request_gpus"; then
  if [[ -n "$slurm_gres" ]]; then
    sbatch_args+=(--gres "$slurm_gres")
  elif [[ -n "$slurm_gpus_per_node" ]]; then
    sbatch_args+=(--gpus-per-node "$slurm_gpus_per_node")
  fi
fi
if [[ -n "${PERF_EVAL_SLURM_ACCOUNT:-}" ]]; then
  sbatch_args+=(--account "$PERF_EVAL_SLURM_ACCOUNT")
fi
if [[ -n "${PERF_EVAL_SLURM_PARTITION:-}" ]]; then
  sbatch_args+=(--partition "$PERF_EVAL_SLURM_PARTITION")
fi
if [[ -n "${PERF_EVAL_SLURM_QOS:-}" ]]; then
  sbatch_args+=(--qos "$PERF_EVAL_SLURM_QOS")
fi
if [[ -n "${PERF_EVAL_SLURM_RESERVATION:-}" ]]; then
  sbatch_args+=(--reservation "$PERF_EVAL_SLURM_RESERVATION")
fi
if [[ -n "${PERF_EVAL_SLURM_TIME:-}" ]]; then
  sbatch_args+=(--time "$PERF_EVAL_SLURM_TIME")
fi
if [[ -n "${PERF_EVAL_SLURM_EXTRA_SBATCH_ARGS:-}" ]]; then
  # shellcheck disable=SC2206  # Buildkite env intentionally supplies words.
  extra_sbatch_args=($PERF_EVAL_SLURM_EXTRA_SBATCH_ARGS)
  sbatch_args+=("${extra_sbatch_args[@]}")
fi

job_id="$(sbatch "${sbatch_args[@]}" "$script")"
job_id="${job_id%%;*}"
printf '%s\n' "$job_id"
