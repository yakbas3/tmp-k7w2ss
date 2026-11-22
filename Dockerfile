# Shopping Sessions BENCH DIAGNOSTIC image. Not an entry.
# It writes the real dev answer, then times the engine variants of
# variants.txt on a generated dataset and fails the dev check on purpose
# with marker rows that carry the timings.
# Build context: this directory.
FROM rust:1.100.0-alpine3.22 AS build
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+crt-static -C llvm-args=-x86-branches-within-32B-boundaries"

# The contest's generator, fetched at build time (public file, pinned hash).
WORKDIR /gen
RUN wget -q -O generator.zip https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip \
 && echo "78f973f75d14b48e792628284f19c75de71fa597d20952e16a9ced93c78bee4a  generator.zip" | sha256sum -c - \
 && unzip -q generator.zip \
 && cd generator \
 && cargo build --release --locked --target x86_64-unknown-linux-musl \
 && cp target/x86_64-unknown-linux-musl/release/generate /generate

# The driver.
WORKDIR /driver
COPY driver/Cargo.toml ./
COPY driver/src ./src
RUN cargo build --release --target x86_64-unknown-linux-musl \
 && cp target/x86_64-unknown-linux-musl/release/driver /driver-bin

# The engine variants named by `build` lines.
WORKDIR /bench
COPY variants ./variants
COPY variants.txt variants-test.txt build_variants.sh ./
RUN sh build_variants.sh variants.txt /bench/variants /out/bin

FROM scratch
COPY --from=build /out/bin /bench/bin
COPY --from=build /generate /bench/generate
COPY --from=build /driver-bin /bench/driver
COPY variants.txt variants-test.txt /bench/
USER 1000:1000
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/bench/driver"]
