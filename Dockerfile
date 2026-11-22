# Shopping Sessions entry: one static binary in an empty image.
# The scorer is linux/amd64 (Cascade Lake or newer), so the build targets
# x86-64-v3 (AVX2, BMI2). The build stage is pinned; the run stage is scratch.
FROM rust:1.100.0-alpine3.22 AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+crt-static -C llvm-args=-x86-branches-within-32B-boundaries"
RUN cargo build --release --locked --target x86_64-unknown-linux-musl \
 && cp target/x86_64-unknown-linux-musl/release/sessions /sessions

FROM scratch
COPY --from=build /sessions /sessions
USER 1000:1000
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/sessions"]
