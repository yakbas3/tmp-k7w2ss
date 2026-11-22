#!/bin/sh
# Run the bench image with the scorer's docker flags (as harness/docker_run.sh)
# plus BENCH_* overrides for a small local test.
#   IMAGE=ss-bench:1 run_local.sh DATA OUT [VAR=VAL ..]
# Example: run_local.sh dev-copy out BENCH_EVENTS=1000000 BENCH_CLICKS=250000
# Prints the container's stderr; the scratch dir is a fresh temp dir.
set -eu
: "${IMAGE:?set IMAGE}"
DATA=$(cd "$1" && pwd); mkdir -p "$2"; OUT=$(cd "$2" && pwd); shift 2
SCRATCH=$(mktemp -d)
chmod 777 "$OUT" "$SCRATCH"
ENVS=""
for kv in "$@"; do ENVS="$ENVS -e $kv"; done
T0=$(date +%s)
set +e
# shellcheck disable=SC2086
docker run --rm --platform linux/amd64 \
  --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 4096 --cpus "${CPUS:-8}" --memory "${MEM:-6g}" --memory-swap "${MEM:-6g}" \
  --user 1000:1000 \
  -v "$DATA":/data:ro -v "$OUT":/output -v "$SCRATCH":/scratch \
  -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch -e HOME=/scratch \
  -e NCPU="${CPUS:-8}" -e MEM_GB="$(echo "${MEM:-6g}" | tr -d g)" $ENVS \
  "$IMAGE"
RC=$?
echo "container rc=$RC wall_s=$(( $(date +%s) - T0 ))" >&2
echo "scratch left: $(ls -A "$SCRATCH" | tr '\n' ' ')" >&2
rm -rf "$SCRATCH"
exit $RC
