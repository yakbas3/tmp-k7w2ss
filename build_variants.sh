#!/bin/sh
# Build every `build` line of the variant table, in parallel, with the real
# Dockerfile's flags. Runs inside the build stage.
#   build_variants.sh TABLE SRC_ROOT OUT_DIR
set -eu
TABLE=$1 SRC=$2 OUT=$3
BASE_FLAGS="-C target-cpu=x86-64-v3 -C target-feature=+crt-static -C llvm-args=-x86-branches-within-32B-boundaries"
mkdir -p "$OUT" /tmp/blogs
pids=""
names=""
grep -E '^[[:space:]]*build[[:space:]]' "$TABLE" | while read -r _ name tree rest; do
  echo "$name $tree $rest"
done > /tmp/builds.txt
while read -r name tree rest; do
  flags="$BASE_FLAGS"
  envs=""
  for kv in $rest; do
    case "$kv" in
      cfg=*) flags="$flags --cfg ${kv#cfg=}" ;;
      *=*) envs="$envs $kv" ;;
      *) echo "bad build item: $kv"; exit 1 ;;
    esac
  done
  [ -d "$SRC/$tree" ] || { echo "no tree $SRC/$tree for $name"; exit 1; }
  echo "build $name: tree=$tree env=[$envs ] RUSTFLAGS=[$flags]"
  (
    cd "$SRC/$tree"
    env $envs RUSTFLAGS="$flags" CARGO_TARGET_DIR="/tmp/target-$name" \
      cargo build --release --locked --target x86_64-unknown-linux-musl \
      > "/tmp/blogs/$name.log" 2>&1 \
      && cp "/tmp/target-$name/x86_64-unknown-linux-musl/release/sessions" "$OUT/$name"
  ) &
  pids="$pids $!"
  names="$names $name"
done < /tmp/builds.txt
fail=0
set -- $names
for p in $pids; do
  if ! wait "$p"; then
    echo "BUILD FAILED: $1"; tail -30 "/tmp/blogs/$1.log"; fail=1
  fi
  shift
done
ls -la "$OUT"
exit $fail
