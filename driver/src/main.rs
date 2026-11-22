//! Benchmark driver for the Shopping Sessions scorer.
//!
//! Run as the container's command, it:
//! 1. runs the first variant on the real input (`$DATA_DIR` into
//!    `$OUTPUT_DIR`), so the dev check sees the right answer;
//! 2. generates a dataset in `$SCRATCH_DIR/data` with the contest generator;
//! 3. runs every variant of the table once to warm up, then `REPS` rounds,
//!    the order rotated by one each round;
//! 4. reports each run and a summary on stderr, and the same as marker rows
//!    `~bench00`.. in `user_daily/part-bench.parquet`, so the dev check
//!    fails on purpose and returns them.
//!
//! A watchdog ends the run at `HARD_S` seconds and still reports.
//! With `BENCH_FAKE` set it is not the driver but a fake engine for tests:
//! `crash` (abort), `fail` (exit 3) or `wrong` (a wrong self-check line).

mod pq;

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------- knobs
// Each can be overridden by `BENCH_<NAME>` for local tests.

/// First pass: events and clicks to generate.
const EVENTS: u64 = 20_000_000;
const CLICKS: u64 = 5_000_000;
/// Optional second pass, run only if the estimate fits the soft budget.
/// 0 disables it.
const EVENTS2: u64 = 60_000_000;
const CLICKS2: u64 = 15_000_000;
/// Measured rounds per pass (after one warm-up run of each variant).
const REPS: usize = 3;
const REPS2: usize = 2;
/// Generator seed; not 1, the dev set's.
const SEED: u64 = 7;
/// No new round or pass starts if its estimate ends past this.
const SOFT_S: f64 = 100.0;
/// The watchdog ends everything here (never more than 145).
const HARD_S: f64 = 140.0;
/// Flush the generated data to disk before timing (no writeback during runs).
const SYNC: u64 = 1;

const TABLE: &str = "/bench/variants.txt";
const BIN_DIR: &str = "/bench/bin";
const GENERATE: &str = "/bench/generate";

fn knob_u64(name: &str, default: u64) -> u64 {
    std::env::var(format!("BENCH_{name}")).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn knob_f64(name: &str, default: f64) -> f64 {
    std::env::var(format!("BENCH_{name}")).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

// ---------------------------------------------------------------- libc

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Timeval {
    sec: i64,
    usec: i64,
}

#[repr(C)]
struct Rusage {
    utime: Timeval,
    stime: Timeval,
    maxrss: i64,
    ixrss: i64,
    idrss: i64,
    isrss: i64,
    minflt: i64,
    majflt: i64,
    nswap: i64,
    inblock: i64,
    oublock: i64,
    rest: [i64; 6],
    reserved: [i64; 16],
}

unsafe extern "C" {
    fn wait4(pid: i32, status: *mut i32, options: i32, ru: *mut Rusage) -> i32;
    fn kill(pid: i32, sig: i32) -> i32;
    fn syncfs(fd: i32) -> i32;
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn mincore(addr: *mut u8, len: usize, vec: *mut u8) -> i32;
    fn _exit(code: i32) -> !;
}

// ---------------------------------------------------------------- state

static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
static CHILD: AtomicI32 = AtomicI32::new(0);
static STOP: AtomicBool = AtomicBool::new(false);
static FINISHING: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<State> = Mutex::new(State::new());

fn epoch_s() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

fn now_s() -> f64 {
    START.get().map_or(0.0, |s| s.elapsed().as_secs_f64())
}

/// What a row is. `Proto`: a prototype whose answer may be wrong; its
/// self-check is compared with the reference and reported, never marked
/// DIFF or bad. `Probe`: writes no output and prints no self-check; only its
/// wall and parse timer count, and a missing self-check is not bad.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Normal,
    Proto,
    Probe,
}

#[derive(Clone)]
struct Variant {
    name: String,
    kind: Kind,
    /// Drop the input from the page cache before every run of this row.
    cold: bool,
    bin: PathBuf,
    env: Vec<(String, String)>,
    /// Binary and env as written, to find identical controls.
    spec: String,
}

/// One process run.
#[derive(Clone, Default)]
struct Proc {
    wall_us: u64,
    user_ms: f64,
    sys_ms: f64,
    rss_mb: u64,
    /// `ok`, `exitN`, `sigN`, `spawn:..`, `killed`, `nocheck`, `DIFF`.
    status: String,
    stderr: String,
    /// From the child's rusage.
    minflt: u64,
    majflt: u64,
    inblock: u64,
    /// Wall clock (unix seconds) just before spawn and just after wait returned.
    fork_epoch: f64,
    wait_epoch: f64,
    /// AnonHugePages of the child at about 300 ms (kB), NaN if not sampled.
    thp_kb: f64,
    /// Cold rows: percent of the input still cached after the drop, NaN otherwise
    /// (negative when it could not be verified).
    cached_pct: f64,
}

#[derive(Clone)]
struct Run {
    pass: usize,
    variant: usize,
    /// -1 is the warm-up.
    rep: i32,
    p: Proc,
    phases: Vec<(String, f64)>,
    check: String,
    out: String,
}

struct Pass {
    events: u64,
    clicks: u64,
    genp: Option<Proc>,
    sync_ms: f64,
    rounds_done: usize,
    note: String,
}

struct State {
    env: Vec<(String, String)>,
    variants: Vec<Variant>,
    dev: Option<Run>,
    passes: Vec<Pass>,
    runs: Vec<Run>,
    notes: Vec<String>,
    out_dir: PathBuf,
    day: i32,
    /// Variant indices of the `first` table line: one run each, right after
    /// generation, before anything else.
    first: Vec<usize>,
    /// The `scatter NAME N` table line: N single runs of that row, each
    /// after a churn, listed one by one.
    scatter: Option<(usize, usize)>,
}

impl State {
    const fn new() -> Self {
        State {
            env: Vec::new(),
            variants: Vec::new(),
            dev: None,
            passes: Vec::new(),
            runs: Vec::new(),
            notes: Vec::new(),
            out_dir: PathBuf::new(),
            day: 0,
            first: Vec::new(),
            scatter: None,
        }
    }
}

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn note(s: String) {
    eprintln!("BENCHNOTE {s}");
    state().notes.push(s);
}

// ---------------------------------------------------------------- helpers

fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
const FNV0: u64 = 0xcbf29ce484222325;

/// Word-at-a-time hash for output files (not cryptographic).
fn hash_words(bytes: &[u8], mut h: u64) -> u64 {
    let mut it = bytes.chunks_exact(8);
    for c in &mut it {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h = (h ^ w).wrapping_mul(0x9E3779B97F4A7C15).rotate_left(29);
    }
    fnv(it.remainder(), h)
}

fn read_trim(p: &str) -> String {
    fs::read_to_string(p).map(|s| s.trim().to_string()).unwrap_or_else(|_| "-".into())
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// First day of the window in the input's manifest, as days since 1970.
fn window_day(data: &Path) -> i32 {
    let Ok(m) = fs::read_to_string(data.join("manifest.json")) else { return 0 };
    let Some(at) = m.find("window_start") else { return 0 };
    let rest = &m[at..];
    let Some(q) = rest.find(": \"").map(|i| i + 3).or_else(|| rest.find(":\"").map(|i| i + 2)) else { return 0 };
    let s = &rest[q..];
    if s.len() < 10 {
        return 0;
    }
    let num = |a: usize, b: usize| s[a..b].parse::<i64>().unwrap_or(0);
    days_from_civil(num(0, 4), num(5, 7), num(8, 10)) as i32
}

// ---------------------------------------------------------------- processes

/// Spawn `cmd` with stdout and stderr to `log`, wait for it with wait4.
/// Wall time is from just before spawn to the return of wait4.
fn run_proc(mut cmd: Command, log: &Path) -> Proc {
    let mut p = Proc::default();
    if STOP.load(Ordering::SeqCst) {
        p.status = "killed".into();
        return p;
    }
    let f = match File::create(log) {
        Ok(f) => f,
        Err(e) => {
            p.status = format!("log:{e}");
            return p;
        }
    };
    let f2 = f.try_clone().expect("dup");
    cmd.stdin(Stdio::null()).stdout(Stdio::from(f2)).stderr(Stdio::from(f));
    p.thp_kb = f64::NAN;
    p.cached_pct = f64::NAN;
    p.fork_epoch = epoch_s();
    let t0 = Instant::now();
    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            p.wall_us = t0.elapsed().as_micros() as u64;
            p.status = format!("spawn:{:?}", e.kind());
            p.stderr = e.to_string();
            return p;
        }
    };
    let pid = child.id() as i32;
    CHILD.store(pid, Ordering::SeqCst);
    // Best effort: AnonHugePages of the child at about 300 ms. Not joined;
    // a run shorter than that has no sample.
    let thp_cell = std::sync::Arc::new(Mutex::new(f64::NAN));
    let running = std::sync::Arc::new(AtomicBool::new(true));
    {
        let (cell, running) = (thp_cell.clone(), running.clone());
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            if running.load(Ordering::SeqCst) {
                if let Ok(t) = fs::read_to_string(format!("/proc/{pid}/smaps_rollup")) {
                    if let Some(l) = t.lines().find(|l| l.starts_with("AnonHugePages:")) {
                        if let Some(v) = l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()) {
                            *cell.lock().unwrap() = v;
                        }
                    }
                }
            }
        });
    }
    let mut st: i32 = 0;
    let mut ru: Rusage = unsafe { std::mem::zeroed() };
    loop {
        let r = unsafe { wait4(pid, &mut st, 0, &mut ru) };
        if r == pid {
            break;
        }
        if r < 0 && std::io::Error::last_os_error().raw_os_error() != Some(4) {
            st = -1;
            break;
        }
    }
    p.wall_us = t0.elapsed().as_micros() as u64;
    p.wait_epoch = epoch_s();
    running.store(false, Ordering::SeqCst);
    p.thp_kb = *thp_cell.lock().unwrap();
    CHILD.store(0, Ordering::SeqCst);
    // Reaped above: std must not wait for it again.
    std::mem::forget(child);
    p.user_ms = ru.utime.sec as f64 * 1e3 + ru.utime.usec as f64 / 1e3;
    p.sys_ms = ru.stime.sec as f64 * 1e3 + ru.stime.usec as f64 / 1e3;
    p.rss_mb = (ru.maxrss as u64) >> 10;
    p.minflt = ru.minflt as u64;
    p.majflt = ru.majflt as u64;
    p.inblock = ru.inblock as u64;
    p.status = if st == -1 {
        "waitfail".into()
    } else if st & 0x7f == 0 {
        let code = (st >> 8) & 0xff;
        if code == 0 { "ok".into() } else { format!("exit{code}") }
    } else {
        format!("sig{}", st & 0x7f)
    };
    if STOP.load(Ordering::SeqCst) && p.status != "ok" {
        p.status = format!("killed({})", p.status);
    }
    let mut s = Vec::new();
    if let Ok(mut f) = File::open(log) {
        let _ = f.read_to_end(&mut s);
    }
    p.stderr = String::from_utf8_lossy(&s).into_owned();
    p
}

/// The child gets the container's own environment, as the scorer gave it
/// to this image (the engine image would get the same), minus BENCH_* knobs.
/// The caller then sets DATA_DIR, OUTPUT_DIR, ENGINE_TIMING and the row's env.
fn base_env(cmd: &mut Command) {
    cmd.env_clear();
    for (k, v) in std::env::vars() {
        if !k.starts_with("BENCH_") {
            cmd.env(k, v);
        }
    }
}

/// `[engine] NAME wall X ms ...` lines, and a hash of the self-check line.
fn parse_engine(stderr: &str) -> (Vec<(String, f64)>, String) {
    let mut phases = Vec::new();
    let mut check = "-".to_string();
    for line in stderr.lines() {
        let Some(rest) = line.strip_prefix("[engine] ") else { continue };
        if let Some(c) = rest.strip_prefix("self-check ") {
            check = format!("{:08x}", fnv(c.trim().as_bytes(), FNV0) as u32);
            continue;
        }
        if let Some(x) = rest.strip_prefix("start epoch ") {
            if let Ok(v) = x.trim().parse::<f64>() {
                phases.push(("start_epoch".into(), v));
            }
            continue;
        }
        if let Some(x) = rest.strip_prefix("exit at ") {
            // "exit at 4169.9 ms, epoch 1795285773.094, peak rss .."
            let t: Vec<&str> = x.split(|c: char| c == ' ' || c == ',').filter(|w| !w.is_empty()).collect();
            if let Some(v) = t.first().and_then(|v| v.parse::<f64>().ok()) {
                phases.push(("exit_ms".into(), v));
            }
            if let Some(i) = t.iter().position(|w| *w == "epoch") {
                if let Some(v) = t.get(i + 1).and_then(|v| v.parse::<f64>().ok()) {
                    phases.push(("exit_epoch".into(), v));
                }
            }
            continue;
        }
        let t: Vec<&str> = rest.split_whitespace().collect();
        if t.len() >= 3 && t[1] == "wall" {
            if let Ok(v) = t[2].parse::<f64>() {
                phases.push((t[0].to_string(), v));
            }
        }
    }
    (phases, check)
}

fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join(" | ")
}

fn hash_dir(dir: &Path) -> (u64, u64) {
    let mut files = Vec::new();
    for sub in ["user_daily", "campaign_daily"] {
        if let Ok(rd) = fs::read_dir(dir.join(sub)) {
            for e in rd.flatten() {
                files.push(e.path());
            }
        }
    }
    files.sort();
    let mut h = FNV0;
    let mut bytes = 0;
    for f in files {
        h = fnv(f.file_name().map_or(&[][..], |n| n.as_encoded_bytes()), h);
        if let Ok(b) = fs::read(&f) {
            bytes += b.len() as u64;
            h = hash_words(&b, h);
        }
    }
    (h, bytes)
}

/// Map, write-touch (one byte per 4 KB page, plain pages, no madvise) and
/// free `CHURN_MB` of anonymous memory, so the next child starts on memory
/// that was just freed rather than on whatever the last child left.
const CHURN_MB: u64 = 1024;

fn churn() {
    let mb = knob_u64("CHURN_MB", CHURN_MB) as usize;
    if mb == 0 {
        return;
    }
    let len = mb << 20;
    // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS
    let addr = unsafe { mmap(std::ptr::null_mut(), len, 3, 0x22, -1, 0) };
    if addr as isize == -1 {
        return;
    }
    let mut i = 0;
    while i < len {
        unsafe { std::ptr::write_volatile(addr.add(i), 1u8) };
        i += 4096;
    }
    unsafe { munmap(addr, len) };
}

/// The `scatter NAME N` line of the table.
fn load_scatter(path: &str, variants: &[Variant]) -> Option<(usize, usize)> {
    let text = fs::read_to_string(path).unwrap_or_default();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() == 3 && t[0] == "scatter" {
            let vi = variants.iter().position(|v| v.name == t[1]);
            let n = t[2].parse::<usize>().ok();
            if let (Some(vi), Some(n)) = (vi, n) {
                return Some((vi, n.min(50)));
            }
            note(format!("scatter: bad line {line}"));
        }
    }
    None
}

/// The `first NAME ..` line of the table, as variant indices.
fn load_first(path: &str, variants: &[Variant]) -> Vec<usize> {
    let text = fs::read_to_string(path).unwrap_or_default();
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.first() == Some(&"first") {
            for n in &t[1..] {
                match variants.iter().position(|v| v.name == *n) {
                    Some(i) => out.push(i),
                    None => note(format!("first: no row {n}")),
                }
            }
        }
    }
    out
}

fn files_under(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            match e.file_type() {
                Ok(t) if t.is_dir() => files_under(&p, out),
                Ok(t) if t.is_file() => out.push(p),
                _ => {}
            }
        }
    }
}

/// Cold rows: syncfs the generated set, then POSIX_FADV_DONTNEED on every
/// input file, then count what is still resident with mmap + mincore.
/// Returns the resident percent, or -1 when it could not be verified.
fn drop_cache(data: &Path) -> f64 {
    use std::os::fd::AsRawFd;
    if let Ok(f) = File::open(data) {
        unsafe { syncfs(f.as_raw_fd()) };
    }
    let mut files = Vec::new();
    files_under(data, &mut files);
    let (mut total, mut resident, mut verified) = (0u64, 0u64, true);
    for path in files {
        let Ok(f) = File::open(&path) else { continue };
        let fd = f.as_raw_fd();
        unsafe { posix_fadvise(fd, 0, 0, 4) };
        let len = f.metadata().map_or(0, |m| m.len()) as usize;
        if len == 0 {
            continue;
        }
        total += len as u64;
        let addr = unsafe { mmap(std::ptr::null_mut(), len, 1, 1, fd, 0) };
        if addr as isize == -1 {
            verified = false;
            continue;
        }
        let pages = len.div_ceil(4096);
        let mut vec = vec![0u8; pages];
        if unsafe { mincore(addr, len, vec.as_mut_ptr()) } == 0 {
            resident += vec.iter().filter(|&&b| b & 1 != 0).count() as u64 * 4096;
        } else {
            verified = false;
        }
        unsafe { munmap(addr, len) };
    }
    if !verified || total == 0 {
        -1.0
    } else {
        100.0 * resident.min(total) as f64 / total as f64
    }
}

fn run_variant(v: &Variant, data: &Path, out: &Path, log: &Path) -> (Proc, Vec<(String, f64)>, String, String) {
    let _ = fs::remove_dir_all(out);
    let cached = if v.cold { drop_cache(data) } else { f64::NAN };
    let mut cmd = Command::new(&v.bin);
    base_env(&mut cmd);
    cmd.env("DATA_DIR", data).env("OUTPUT_DIR", out).env("ENGINE_TIMING", "1");
    for (k, val) in &v.env {
        cmd.env(k, val);
    }
    let mut p = run_proc(cmd, log);
    p.cached_pct = cached;
    let (phases, check) = parse_engine(&p.stderr);
    if p.status == "ok" && check == "-" && v.kind != Kind::Probe {
        p.status = "nocheck".into();
    }
    let (h, bytes) = hash_dir(out);
    let outh = if bytes == 0 { "-".to_string() } else { format!("{:08x}", h as u32) };
    let _ = fs::remove_dir_all(out);
    (p, phases, check, outh)
}

// ---------------------------------------------------------------- table

fn load_table(path: &str) -> Result<Vec<Variant>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out: Vec<Variant> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() || t[0] == "build" || t[0] == "first" || t[0] == "scatter" {
            continue;
        }
        if t[0] != "run" || t.len() < 3 {
            return Err(format!("{path}:{}: expected `run NAME BIN [K=V ..]`", i + 1));
        }
        let bin = if t[2] == "@self" {
            std::env::current_exe().unwrap_or_else(|_| PathBuf::from("/bench/driver"))
        } else if t[2].starts_with('/') {
            PathBuf::from(t[2])
        } else {
            Path::new(BIN_DIR).join(t[2])
        };
        let mut env = Vec::new();
        let mut kind = Kind::Normal;
        let mut cold = false;
        for kv in &t[3..] {
            match *kv {
                "cold=1" => {
                    cold = true;
                    continue;
                }
                "kind=proto" => {
                    kind = Kind::Proto;
                    continue;
                }
                "kind=probe" => {
                    kind = Kind::Probe;
                    continue;
                }
                "kind=normal" => continue,
                _ => {}
            }
            let Some((k, v)) = kv.split_once('=') else {
                return Err(format!("{path}:{}: bad env `{kv}`", i + 1));
            };
            env.push((k.to_string(), v.to_string()));
        }
        if out.iter().any(|v| v.name == t[1]) {
            return Err(format!("{path}:{}: duplicate name {}", i + 1, t[1]));
        }
        out.push(Variant { name: t[1].to_string(), kind, cold, bin, env, spec: t[2..].join(" ") });
    }
    if out.is_empty() {
        return Err(format!("{path}: no run lines"));
    }
    Ok(out)
}

// ---------------------------------------------------------------- environment

fn environment() -> Vec<(String, String)> {
    let mut e = Vec::new();
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let field = |k: &str| {
        cpuinfo
            .lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_once(':'))
            .map_or("-".to_string(), |(_, v)| v.trim().to_string())
    };
    e.push(("cpu".into(), field("model name")));
    let flags = field("flags");
    let has = |f: &str| if flags.split_whitespace().any(|x| x == f) { "1" } else { "0" };
    e.push((
        "avx512".into(),
        format!("f{} bw{} vl{} vbmi2{}", has("avx512f"), has("avx512bw"), has("avx512vl"), has("avx512_vbmi2")),
    ));
    e.push(("nproc".into(), std::thread::available_parallelism().map_or(0, |n| n.get()).to_string()));
    e.push(("kernel".into(), read_trim("/proc/sys/kernel/osrelease")));
    let thp = |p: &str| {
        let s = read_trim(p);
        s.split_whitespace().find(|w| w.starts_with('[')).map_or(s.clone(), |w| w.trim_matches(['[', ']']).to_string())
    };
    e.push(("thp".into(), thp("/sys/kernel/mm/transparent_hugepage/enabled")));
    e.push(("thpdefrag".into(), thp("/sys/kernel/mm/transparent_hugepage/defrag")));
    e.push(("cpumax".into(), read_trim("/sys/fs/cgroup/cpu.max")));
    e.push(("cpuset".into(), read_trim("/sys/fs/cgroup/cpuset.cpus.effective")));
    let mem = read_trim("/sys/fs/cgroup/memory.max");
    let mem = mem.parse::<u64>().map_or(mem.clone(), |b| format!("{}M", b >> 20));
    e.push(("memmax".into(), mem));
    e.push(("pids".into(), read_trim("/sys/fs/cgroup/pids.max")));
    e
}

// ---------------------------------------------------------------- report

fn fmt_ms(v: Option<f64>) -> String {
    v.map_or("-".into(), |x| format!("{x:.1}"))
}

fn phase(r: &Run, name: &str) -> Option<f64> {
    r.phases.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
}

fn bench_line(st: &State, r: &Run) -> String {
    let v = &st.variants[r.variant];
    let rep = if r.rep <= -100 {
        format!("s{}", -100 - r.rep)
    } else if r.rep <= -10 {
        format!("f{}", -10 - r.rep)
    } else if r.rep < 0 {
        "w".to_string()
    } else {
        r.rep.to_string()
    };
    format!(
        "BENCH {} {} {} {} {} {} {:.0} {:.0} {} {} {} out={} pass={}",
        v.name,
        rep,
        r.p.wall_us,
        fmt_ms(phase(r, "parse")),
        fmt_ms(phase(r, "resolve")),
        fmt_ms(phase(r, "person")),
        r.p.user_ms,
        r.p.sys_ms,
        r.p.rss_mb,
        r.check,
        r.p.status,
        r.out,
        r.pass + 1
    )
}

/// Per-run metrics, in this order (NaN when not available).
const M_WALL: usize = 0; // process wall, fork to wait return, ms
const M_EXIT: usize = 1; // the engine's own `exit at`, ms after its start
const M_GSTART: usize = 2; // engine start epoch minus fork epoch, ms
const M_GEXIT: usize = 3; // wait return minus the engine's exit epoch, ms
const M_SYS: usize = 4; // system CPU, ms
const M_MINF: usize = 5; // minor faults
const M_MAJF: usize = 6; // major faults
const M_RSS: usize = 7; // peak RSS, MB
const M_THP: usize = 8; // AnonHugePages at ~300 ms, kB
const M_CACHED: usize = 9; // cold rows: percent still cached after the drop
const M_USER: usize = 10; // user CPU, ms
const M_INBLK: usize = 11; // blocks read
const NM: usize = 12;

fn metrics(r: &Run) -> [f64; NM] {
    let ph = |n: &str| phase(r, n).unwrap_or(f64::NAN);
    let mut m = [f64::NAN; NM];
    m[M_WALL] = r.p.wall_us as f64 / 1e3;
    m[M_EXIT] = ph("exit_ms");
    m[M_GSTART] = (ph("start_epoch") - r.p.fork_epoch) * 1e3;
    m[M_GEXIT] = (r.p.wait_epoch - ph("exit_epoch")) * 1e3;
    m[M_SYS] = r.p.sys_ms;
    m[M_MINF] = r.p.minflt as f64;
    m[M_MAJF] = r.p.majflt as f64;
    m[M_RSS] = r.p.rss_mb as f64;
    m[M_THP] = r.p.thp_kb;
    m[M_CACHED] = r.p.cached_pct;
    m[M_USER] = r.p.user_ms;
    m[M_INBLK] = r.p.inblock as f64;
    m
}

fn nanmedian(v: impl Iterator<Item = f64>) -> f64 {
    let mut x: Vec<f64> = v.filter(|x| !x.is_nan()).collect();
    median(&mut x)
}

struct Summary {
    /// Medians of `metrics` over the measured runs.
    m: [f64; NM],
    name: String,
    n: usize,
    bad: usize,
    med: f64,
    min: f64,
    max: f64,
    /// Paired differences against variant 0, ms: median, min, max, count.
    d: Option<(f64, f64, f64, usize)>,
    /// parse, resolve, person, convert (ms medians; NaN when not printed).
    phases: [f64; 4],
    user: f64,
    sys: f64,
    rss: f64,
    check: String,
    status: String,
}

fn summarize(st: &State, pass: usize) -> Vec<Summary> {
    let runs: Vec<&Run> = st.runs.iter().filter(|r| r.pass == pass).collect();
    let wall_of = |v: usize, rep: i32| {
        runs.iter().find(|r| r.variant == v && r.rep == rep && timed(r)).map(|r| r.p.wall_us as f64 / 1e3)
    };
    let mut out = Vec::new();
    for (vi, v) in st.variants.iter().enumerate() {
        let mine: Vec<&&Run> = runs.iter().filter(|r| r.variant == vi && r.rep >= 0).collect();
        let all: Vec<&&Run> = runs.iter().filter(|r| r.variant == vi).collect();
        let ok: Vec<&&Run> = mine.iter().copied().filter(|r| timed(r)).collect();
        let mut walls: Vec<f64> = ok.iter().map(|r| r.p.wall_us as f64 / 1e3).collect();
        let med = median(&mut walls);
        let mut d = None;
        if vi > 0 {
            let mut diffs = Vec::new();
            for r in &ok {
                if let Some(base) = wall_of(0, r.rep) {
                    diffs.push(r.p.wall_us as f64 / 1e3 - base);
                }
            }
            if !diffs.is_empty() {
                let mn = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let c = diffs.len();
                d = Some((median(&mut diffs), mn, mx, c));
            }
        }
        let pm = |name: &str| {
            let mut x: Vec<f64> = ok.iter().filter_map(|r| phase(r, name)).collect();
            median(&mut x)
        };
        let mut user: Vec<f64> = ok.iter().map(|r| r.p.user_ms).collect();
        let mut sys: Vec<f64> = ok.iter().map(|r| r.p.sys_ms).collect();
        let mut rss: Vec<f64> = ok.iter().map(|r| r.p.rss_mb as f64).collect();
        let mut checks: Vec<String> = all.iter().map(|r| r.check.clone()).collect();
        checks.dedup();
        let bad: Vec<&&Run> = all.iter().copied().filter(|r| r.p.status != "ok").collect();
        let status = if all.is_empty() {
            "notrun".to_string()
        } else if bad.is_empty() {
            "ok".to_string()
        } else {
            let mut s: Vec<String> = bad.iter().map(|r| r.p.status.clone()).collect();
            s.dedup();
            format!("{}/{}:{}", bad.len(), all.len(), s.join(","))
        };
        let mut m = [f64::NAN; NM];
        for (k, slot) in m.iter_mut().enumerate() {
            *slot = nanmedian(ok.iter().map(|r| metrics(r)[k]));
        }
        out.push(Summary {
            m,
            name: v.name.clone(),
            n: walls.len(),
            bad: bad.len(),
            med,
            min: walls.first().copied().unwrap_or(f64::NAN),
            max: walls.last().copied().unwrap_or(f64::NAN),
            d,
            phases: [pm("parse"), pm("resolve"), pm("person"), pm("convert")],
            user: median(&mut user),
            sys: median(&mut sys),
            rss: median(&mut rss),
            check: checks.join("/"),
            status,
        });
    }
    out
}

/// The contest site shows only the first three marker rows (about 2000
/// characters) and may turn `=`, `&`, `?` and `;` into spaces, so these rows
/// use none of them; `|` separates fields. Each row stays within ROW_MAX.
/// Row 0, most important first: size, rounds, generator and sync; SCATTER:
///   single runs of one row, each after the churn, listed one by one as
///   wall/sys ms/minor faults, then median, min, max and any run 100 ms or
///   more above the median of the others (SLOW); WARM: the first row's
///   median wall [min-max] and p/r/s, the ROW0_VS rows against it with
///   their p/r/s; the paired deltas of the other rows (cold rows add the
///   percent still cached after the drop); the very first engine run on the
///   generated data (w wall, e the engine's exit-at, gs/gx the gaps at start
///   and at exit, s sys, f minor faults, M major faults); the answer check.
/// Row 1: every row's medians: wall, exit-at, gaps, user, sys, faults,
///   AnonHugePages (MB) at ~300 ms, p/r/s. RSS and blocks read are in later rows.
/// Row 2: the first runs' extra fields, then wall/exit-at of every run in
///   order (s scatter, f first, w warm-up).
const ROW_MAX: usize = 650;
/// Rows compared head to head with the first row in row 0 (with p/r/s).
const ROW0_VS: &[&str] = &["B3"];

fn clean_row(r: String) -> String {
    let mut r: String = r.chars().map(|c| if matches!(c, '=' | '&' | '?' | ';') { ':' } else { c }).collect();
    if r.len() > ROW_MAX {
        let mut cut = ROW_MAX;
        while !r.is_char_boundary(cut) {
            cut -= 1;
        }
        r.truncate(cut);
    }
    r
}

fn top_rows(st: &State, tag: &str, pass: &Pass, sums: &[Summary]) -> [String; 3] {
    let pi = st.passes.iter().position(|p| std::ptr::eq(p, pass)).unwrap_or(0);
    let kind = |vi: usize| st.variants.get(vi).map_or(Kind::Normal, |v| v.kind);
    let f0 = |v: f64| if v.is_nan() { "-".to_string() } else { format!("{v:.0}") };
    let k = |v: f64| if v.is_nan() { "-".to_string() } else { format!("{:.0}k", v / 1e3) };
    // w wall, e engine exit-at, gs/gx start and exit gaps, s sys, f minor faults, M major.
    let full = |label: &str, m: &[f64; NM]| {
        let mut x = format!(
            "|{label} w{} e{} gs{} gx{} s{} f{} M{}",
            f0(m[M_WALL]),
            f0(m[M_EXIT]),
            f0(m[M_GSTART]),
            f0(m[M_GEXIT]),
            f0(m[M_SYS]),
            k(m[M_MINF]),
            f0(m[M_MAJF])
        );
        if !m[M_CACHED].is_nan() {
            x += &if m[M_CACHED] < 0.0 { " c?".to_string() } else { format!(" c{:.1}%", m[M_CACHED]) };
        }
        x
    };
    // Row 0.
    let gen_s = pass.genp.as_ref().map_or("-".into(), |g| {
        let s = format!("{:.1}s", g.wall_us as f64 / 1e6);
        if g.status == "ok" { s } else { format!("{s}!{}", g.status) }
    });
    let mut r0 = format!(
        "{tag} {}M/{}M r{} gen{gen_s} sync{:.1}s",
        pass.events as f64 / 1e6,
        pass.clicks as f64 / 1e6,
        pass.rounds_done,
        pass.sync_ms / 1e3
    );
    if !pass.note.is_empty() {
        let n: String = pass.note.trim().chars().take(50).collect();
        r0 += &format!(" !{n}");
    }
    let churn_mb = knob_u64("CHURN_MB", CHURN_MB);
    let mut firsts: Vec<&Run> = st.runs.iter().filter(|r| r.pass == pi && r.rep <= -10 && r.rep > -100).collect();
    firsts.sort_by_key(|r| -r.rep);
    // 1. Scatter: single runs one by one, wall/sys/minor faults, and the
    //    finding if one is 100 ms or more slower than the median of the rest.
    let mut sc: Vec<&Run> = st.runs.iter().filter(|r| r.pass == pi && r.rep <= -100).collect();
    sc.sort_by_key(|r| -r.rep);
    if let Some(r) = sc.first() {
        let name = st.variants[r.variant].name.as_str();
        let items: Vec<String> = sc
            .iter()
            .map(|r| {
                let m = metrics(r);
                let x = format!("{}/{}/{}", f0(m[M_WALL]), f0(m[M_SYS]), k(m[M_MINF]));
                if r.p.status == "ok" { x } else { format!("{x}!{}", r.p.status) }
            })
            .collect();
        r0 += &format!("|SCATTER {name} x{} churn{churn_mb}M each w/s/f {}", sc.len(), items.join(","));
        let walls: Vec<f64> = sc.iter().map(|r| r.p.wall_us as f64 / 1e3).collect();
        let mut slow: Vec<String> = Vec::new();
        for (i, w) in walls.iter().enumerate() {
            let rest = nanmedian(walls.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, v)| *v));
            if w - rest >= 100.0 {
                slow.push(format!("run{} +{:.0}ms", i + 1, w - rest));
            }
        }
        let mut sorted = walls.clone();
        let med = median(&mut sorted);
        r0 += &format!(
            " med{} min{} max{}{}",
            f0(med),
            f0(sorted.first().copied().unwrap_or(f64::NAN)),
            f0(sorted.last().copied().unwrap_or(f64::NAN)),
            if slow.is_empty() { " no slow run".to_string() } else { format!(" SLOW {}", slow.join(",")) }
        );
    }
    // 2. B3 warm against A warm, p/r/s for both.
    let base = sums.first().map_or(f64::NAN, |s| s.med);
    let prs = |x: &Summary| format!("p/r/s {}/{}/{}", f0(x.phases[0]), f0(x.phases[1]), f0(x.phases[2]));
    if let Some(a) = sums.first() {
        r0 += &format!("|WARM churn{churn_mb}M each|{} w{}[{}-{}] {}", a.name, f0(a.med), f0(a.min), f0(a.max), prs(a));
    }
    for x in sums.iter().skip(1).filter(|x| ROW0_VS.contains(&x.name.as_str())) {
        r0 += &format!("|{} w{}", x.name, f0(x.med));
        if let Some((m, mn, mx, _)) = x.d {
            r0 += &format!(" d{}[{mn:+.0},{mx:+.0}]", pct(m, base));
        }
        r0 += &format!(" {}", prs(x));
    }
    // 3. The deltas of the other rows.
    for (vi, x) in sums.iter().enumerate() {
        if vi > 0 && !ROW0_VS.contains(&x.name.as_str()) {
            r0 += &format!("|{}", x.name);
            if let Some((m, mn, mx, _)) = x.d {
                r0 += &format!(" d{}[{mn:+.0},{mx:+.0}]", pct(m, base));
            }
            if !x.m[M_CACHED].is_nan() {
                r0 += &if x.m[M_CACHED] < 0.0 { " c?".to_string() } else { format!(" c{:.1}%", x.m[M_CACHED]) };
            }
        }
    }
    for x in sums.iter() {
        if x.status != "ok" {
            r0 += &format!(" !{}:{}", x.name, x.status);
        }
    }
    // 4. The very first engine run on the generated data.
    for r in &firsts {
        let mut f = full(&format!("1st {}", st.variants[r.variant].name), &metrics(r));
        if r.p.status != "ok" {
            f += &format!(" !{}", r.p.status);
        }
        r0 += &f;
    }
    let reference = sums.first().map(|s| s.check.clone()).unwrap_or_default();
    let differ: Vec<&str> = sums
        .iter()
        .enumerate()
        .filter(|(vi, x)| kind(*vi) != Kind::Probe && (x.check != reference || x.check == "-" || x.check.contains('/')))
        .map(|(_, x)| x.name.as_str())
        .collect();
    r0 += &if reference == "-" || reference.is_empty() {
        "|ck none".to_string()
    } else if differ.is_empty() {
        "|ck all same".to_string()
    } else {
        format!("|ck DIFF {}", differ.join(","))
    };
    // Row 1: every row's medians in full, then user CPU, RSS, THP and blocks read.
    let mut r1 = format!("{tag} medians");
    for x in sums {
        r1 += &format!(
            "|{} w{} e{} gs{} gx{} u{} s{} f{} M{} thpM{} p/r/s {}/{}/{}",
            x.name,
            f0(x.med),
            f0(x.m[M_EXIT]),
            f0(x.m[M_GSTART]),
            f0(x.m[M_GEXIT]),
            f0(x.m[M_USER]),
            f0(x.m[M_SYS]),
            k(x.m[M_MINF]),
            f0(x.m[M_MAJF]),
            f0(x.m[M_THP] / 1024.0),
            f0(x.phases[0]),
            f0(x.phases[1]),
            f0(x.phases[2])
        );
        if !x.m[M_CACHED].is_nan() {
            r1 += &format!(" c{:.1}%", x.m[M_CACHED]);
        }
    }
    // Row 2: the first runs' extra fields, then raw walls.
    let mut r2 = format!("{tag} first extras");
    for r in &firsts {
        let m = metrics(r);
        r2 += &format!(
            "|1{} u{} rss{} thpM{} inb{}",
            st.variants[r.variant].name,
            f0(m[M_USER]),
            f0(m[M_RSS]),
            f0(m[M_THP] / 1024.0),
            f0(m[M_INBLK])
        );
    }
    // Row 2: raw walls and engine exit-at times in round order (w is the warm-up).
    let walls = |vi: usize| -> String {
        let w: Vec<String> = st
            .runs
            .iter()
            .filter(|r| r.pass == pi && r.variant == vi)
            .map(|r| {
                let m = metrics(r);
                let x = format!("{}/{}", f0(m[M_WALL]), f0(m[M_EXIT]));
                let x = if r.rep <= -100 {
                    format!("s{x}")
                } else if r.rep <= -10 {
                    format!("f{x}")
                } else if r.rep < 0 {
                    format!("w{x}")
                } else {
                    x
                };
                if timed(r) { x } else { format!("{x}!") }
            })
            .collect();
        format!("|{} {}", st.variants[vi].name, w.join(","))
    };
    r2 += "|wall/exit ms";
    for vi in 0..st.variants.len() {
        let w = walls(vi);
        if r2.len() + w.len() <= ROW_MAX {
            r2 += &w;
        }
    }
    [clean_row(r0), clean_row(r1), clean_row(r2)]
}

fn pct(d: f64, base: f64) -> String {
    format!("{:+.1}%", d / base * 100.0)
}

/// Prints the summary and returns the marker row texts.
fn report(why: &str) -> Vec<String> {
    let st = state();
    let mut items: Vec<String> = Vec::new();
    let mut later: Vec<String> = Vec::new();
    let mut top: Option<(usize, [String; 3])> = None;
    let total = now_s();
    eprintln!("BENCHEND why={why} total_s={total:.1}");
    for (pi, pass) in st.passes.iter().enumerate() {
        let tag = format!("P{}", pi + 1);
        let gp = pass.genp.as_ref();
        let gen_s = gp.map_or("-".into(), |g| format!("{:.1}", g.wall_us as f64 / 1e6));
        let gen_st = gp.map_or("-".into(), |g| g.status.clone());
        let head = format!(
            "{tag} ev={}M cl={}M rounds={} gen_s={gen_s}({gen_st}) sync_ms={:.0}{}",
            pass.events as f64 / 1e6,
            pass.clicks as f64 / 1e6,
            pass.rounds_done,
            pass.sync_ms,
            if pass.note.is_empty() { String::new() } else { format!(" {}", pass.note) }
        );
        eprintln!("SUMMARYHEAD {head}");
        let row0_at = items.len();
        items.push(head);
        let mut aa: Option<String> = None;
        let sums = summarize(&st, pi);
        let base = sums.first().map_or(f64::NAN, |s| s.med);
        eprintln!("SUMMARY name n med_ms min_ms max_ms dmed_ms dpct dmin_ms dmax_ms pairs parse resolve person user sys rss check status");
        for (vi, s) in sums.iter().enumerate() {
            let (dm, dpc, dmn, dmx, dc) = match s.d {
                Some((m, mn, mx, c)) => (format!("{m:+.1}"), pct(m, base), format!("{mn:+.1}"), format!("{mx:+.1}"), c),
                None => ("-".into(), "-".into(), "-".into(), "-".into(), 0),
            };
            eprintln!(
                "SUMMARY {} {} {:.1} {:.1} {:.1} {dm} {dpc} {dmn} {dmx} {dc} {:.1} {:.1} {:.1} {:.0} {:.0} {:.0} {} {}",
                s.name, s.n, s.med, s.min, s.max, s.phases[0], s.phases[1], s.phases[2], s.user, s.sys, s.rss, s.check, s.status
            );
            let flag = if s.status == "ok" { String::new() } else { format!(" !{}", s.status) };
            if vi == 0 {
                items.push(format!("{}={:.1}ms[{:.1}-{:.1}]{}", s.name, s.med, s.min, s.max, flag));
            } else {
                items.push(match s.d {
                    Some((m, mn, mx, _)) => format!("{}={}[{:+.0},{:+.0}]{}", s.name, pct(m, base), mn, mx, flag),
                    None => format!("{}=n/a{}", s.name, flag),
                });
            }
            later.push(if s.n == 0 {
                format!("{tag}.{} n=0 bad={} ck={}", s.name, s.bad, s.check)
            } else {
                let f = |x: f64| if x.is_nan() { "-".to_string() } else { format!("{x:.0}") };
                format!(
                    "{tag}.{} p/r/s={}/{}/{} u/s={:.0}/{:.0} rss={:.0} ck={} n={} bad={}",
                    s.name,
                    f(s.phases[0]),
                    f(s.phases[1]),
                    f(s.phases[2]),
                    s.user,
                    s.sys,
                    s.rss,
                    s.check,
                    s.n,
                    s.bad
                )
            });
        }
        // Noise floor: later variants identical to an earlier one.
        for (vi, v) in st.variants.iter().enumerate() {
            if let Some(ci) = st.variants[..vi].iter().position(|c| c.spec == v.spec) {
                if ci == 0 {
                    if let Some((m, mn, mx, c)) = sums[vi].d {
                        let mut abs: Vec<f64> = Vec::new();
                        let runs = |x: usize| st.runs.iter().filter(move |r| r.pass == pi && r.variant == x && r.rep >= 0 && timed(r));
                        for r in runs(vi) {
                            if let Some(b) = runs(0).find(|b| b.rep == r.rep) {
                                abs.push(((r.p.wall_us as f64 - b.p.wall_us as f64) / 1e3).abs());
                            }
                        }
                        let ma = median(&mut abs);
                        let line = format!(
                            "noise {}-{}: med {} |d| {} range {mn:+.1}..{mx:+.1} ms n={c}",
                            v.name,
                            st.variants[0].name,
                            pct(m, base),
                            pct(ma, base)
                        );
                        eprintln!("NOISE {tag} {line}");
                        items.push(line);
                        if aa.is_none() {
                            aa = Some(format!("AA {} med{} abs{} [{mn:+.0},{mx:+.0}] n{c}", v.name, pct(m, base), pct(ma, base)));
                        }
                    }
                } else {
                    eprintln!("NOISE {tag} {} repeats {} (not the first variant; compare their SUMMARY rows)", v.name, st.variants[ci].name);
                }
            }
        }
        let _ = (row0_at, aa);
        let nt = st.runs.iter().filter(|r| r.pass == pi && r.rep >= 0 && timed(r)).count();
        if top.as_ref().is_none_or(|(n, _)| nt > *n) {
            top = Some((nt, top_rows(&st, &tag, pass, &sums)));
        }
    }
    // Environment, generator, dev run.
    let mut envs: Vec<String> = Vec::new();
    for (k, v) in &st.env {
        eprintln!("ENV {k}={v}");
        envs.push(format!("{k}={v}"));
    }
    if let Some(d) = &st.dev {
        let line = format!(
            "dev {} wall={:.1}ms p/r/s={}/{}/{} ck={} {}",
            st.variants[d.variant].name,
            d.p.wall_us as f64 / 1e3,
            fmt_ms(phase(d, "parse")),
            fmt_ms(phase(d, "resolve")),
            fmt_ms(phase(d, "person")),
            d.check,
            d.p.status
        );
        eprintln!("DEV {line}");
        envs.push(line);
    }
    for n in &st.notes {
        envs.push(format!("note={n}"));
    }
    envs.push(format!("psi_mem={}", read_trim("/proc/pressure/memory").replace('\n', " / ")));
    envs.push(format!("psi_cpu={}", read_trim("/proc/pressure/cpu").replace('\n', " / ")));
    envs.push(format!("psi_io={}", read_trim("/proc/pressure/io").replace('\n', " / ")));
    let mut keys: Vec<String> = std::env::vars().map(|(k, _)| k).filter(|k| !k.starts_with("BENCH_")).collect();
    keys.sort();
    envs.push(format!("child_env={}", keys.join(",")));
    envs.push(format!("total_s={total:.1} why={why}"));
    // Raw walls per variant and pass, in round order.
    let mut raws: Vec<String> = Vec::new();
    for pi in 0..st.passes.len() {
        for (vi, v) in st.variants.iter().enumerate() {
            let w: Vec<String> = st
                .runs
                .iter()
                .filter(|r| r.pass == pi && r.variant == vi)
                .map(|r| {
                    let x = format!("{:.1}", r.p.wall_us as f64 / 1e3);
                    if r.p.status == "ok" { x } else { format!("{x}!{}", r.p.status) }
                })
                .collect();
            raws.push(format!("P{}.{}:{}", pi + 1, v.name, w.join(",")));
        }
    }
    // Order: summary, environment (cpu and generator first), per-variant
    // phases, raw walls.
    let mut all = items;
    all.extend(envs);
    all.extend(later);
    all.extend(raws);
    let mut rows: Vec<String> = match top {
        Some((_, t)) => t.to_vec(),
        None => vec!["no pass ran".to_string(), "-".to_string(), "-".to_string()],
    };
    let rest = pack(all);
    rows.extend(rest.into_iter().take(24 - 3));
    rows
}

/// Items joined by `;` into rows of at most 400 bytes, at most 24 rows.
fn pack(items: Vec<String>) -> Vec<String> {
    let mut rows: Vec<String> = Vec::new();
    let mut cur = String::new();
    for mut it in items {
        if it.len() > 390 {
            let mut cut = 390;
            while !it.is_char_boundary(cut) {
                cut -= 1;
            }
            it.truncate(cut);
        }
        if !cur.is_empty() && cur.len() + 1 + it.len() > 400 {
            rows.push(std::mem::take(&mut cur));
            if rows.len() == 24 {
                return rows;
            }
        }
        if !cur.is_empty() {
            cur.push(';');
        }
        cur.push_str(&it);
    }
    if !cur.is_empty() {
        rows.push(cur);
    }
    rows
}

fn write_rows(rows: &[String]) {
    let (out, day) = {
        let st = state();
        (st.out_dir.clone(), st.day)
    };
    let n = rows.len() as i32;
    let data: Vec<(String, i32, i32, i32, i32, String)> =
        rows.iter().enumerate().map(|(i, r)| (format!("~bench{i:02}"), day, i as i32 + 1, n, 0, r.clone())).collect();
    let bytes = pq::user_rows(&data);
    let dir = out.join("user_daily");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("part-bench.parquet");
    let tmp = dir.join(".part-bench.tmp");
    let res = fs::write(&tmp, &bytes).and_then(|_| fs::rename(&tmp, &path));
    match res {
        Ok(()) => eprintln!("BENCHROWS wrote {} marker rows to {}", rows.len(), path.display()),
        Err(e) => eprintln!("BENCHROWS failed to write {}: {e}", path.display()),
    }
}

/// Report, write the rows, delete the generated data and exit. Runs once,
/// from the main thread or from the watchdog.
fn finish(why: &str) -> ! {
    if FINISHING.swap(true, Ordering::SeqCst) {
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    let rows = report(why);
    for (i, r) in rows.iter().enumerate() {
        eprintln!("ROW{i:02} {r}");
    }
    write_rows(&rows);
    let t = Instant::now();
    let _ = fs::remove_dir_all(scratch().join("data"));
    let _ = fs::remove_dir_all(scratch().join("out"));
    eprintln!("BENCHDONE cleanup_ms={:.0} total_s={:.1}", t.elapsed().as_secs_f64() * 1e3, now_s());
    let _ = std::io::stderr().flush();
    unsafe { _exit(0) }
}

fn scratch() -> PathBuf {
    PathBuf::from(std::env::var("SCRATCH_DIR").unwrap_or_else(|_| "/scratch".into()))
}

// ---------------------------------------------------------------- passes

fn generate(events: u64, clicks: u64, data: &Path) -> Proc {
    let _ = fs::remove_dir_all(data);
    let threads = std::env::var("NCPU")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()));
    let mut cmd = Command::new(GENERATE);
    base_env(&mut cmd);
    cmd.args(["--seed", &SEED.to_string(), "--events", &events.to_string(), "--clicks", &clicks.to_string()])
        .arg("--output")
        .arg(data)
        .args(["--threads", &threads.to_string()]);
    let p = run_proc(cmd, &scratch().join("gen.log"));
    eprintln!(
        "GEN events={events} clicks={clicks} threads={threads} seed={SEED} wall_ms={:.0} user_ms={:.0} sys_ms={:.0} rss_mb={} status={}{}",
        p.wall_us as f64 / 1e3,
        p.user_ms,
        p.sys_ms,
        p.rss_mb,
        p.status,
        if p.status == "ok" { String::new() } else { format!(" tail: {}", tail(&p.stderr, 3)) }
    );
    p
}

/// syncfs in a thread; waits at most `SYNC_MAX_S`, then goes on (the
/// rest of the writeback then overlaps the first runs, and the row says so).
const SYNC_MAX_S: f64 = 20.0;

fn sync_data(data: &Path) -> (f64, bool) {
    let t = Instant::now();
    let done = std::sync::Arc::new(AtomicBool::new(false));
    let d2 = done.clone();
    let data = data.to_path_buf();
    std::thread::spawn(move || {
        if let Ok(f) = File::open(&data) {
            use std::os::fd::AsRawFd;
            unsafe { syncfs(f.as_raw_fd()) };
        }
        d2.store(true, Ordering::SeqCst);
    });
    let max = knob_f64("SYNC_MAX_S", SYNC_MAX_S);
    while !done.load(Ordering::SeqCst) && t.elapsed().as_secs_f64() < max && !STOP.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(5));
    }
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let finished = done.load(Ordering::SeqCst);
    eprintln!("SYNC ms={ms:.0} finished={finished}");
    (ms, finished)
}

/// One pass: generate, warm up, `reps` rotated rounds within `soft`.
/// Returns the mean ok wall of a run in seconds (for the next estimate).
fn pass(pi: usize, events: u64, clicks: u64, reps: usize, soft: f64) -> Option<f64> {
    let sc = scratch();
    let data = sc.join("data");
    let out = sc.join("out");
    let log = sc.join("run.log");
    state().passes.push(Pass { events, clicks, genp: None, sync_ms: 0.0, rounds_done: 0, note: String::new() });
    let g = generate(events, clicks, &data);
    let ok = g.status == "ok";
    state().passes[pi].genp = Some(g);
    if !ok {
        state().passes[pi].note = "generator failed".into();
        return None;
    }
    if knob_u64("SYNC", SYNC) != 0 && !STOP.load(Ordering::SeqCst) {
        let (ms, finished) = sync_data(&data);
        let mut st = state();
        st.passes[pi].sync_ms = ms;
        if !finished {
            st.passes[pi].note = "sync unfinished".into();
        }
    }
    let variants = state().variants.clone();
    let n = variants.len();
    let mut reference: Option<String> = None;
    let mut do_run = |vi: usize, rep: i32| {
        let v = &variants[vi];
        let (mut p, phases, check, outh) = run_variant(v, &data, &out, &log);
        // The reference answer is the first ok run: variant 0's warm-up,
        // or, if that failed, the next variant that ran ok.
        if p.status == "ok" && v.kind == Kind::Normal {
            match &reference {
                None => {
                    reference = Some(check.clone());
                    if vi != 0 {
                        note(format!("P{}: reference answer from {} (first variant not ok)", pi + 1, v.name));
                    }
                }
                Some(r) if *r != check => p.status = "DIFF".into(),
                _ => {}
            }
        }
        if p.status != "ok" {
            eprintln!("BENCHFAIL {} rep {rep} {} tail: {}", v.name, p.status, tail(&p.stderr, 4));
        }
        let r = Run { pass: pi, variant: vi, rep, p, phases, check, out: outh };
        let mut st = state();
        eprintln!("{}", bench_line(&st, &r));
        st.runs.push(r);
    };
    // The first row: one run each, in table order of the `first` line, right
    // after generation; churn between them (not before the very first).
    let first = state().first.clone();
    for (i, &vi) in first.iter().enumerate() {
        if STOP.load(Ordering::SeqCst) {
            break;
        }
        if i > 0 {
            churn();
        }
        do_run(vi, -10 - i as i32);
    }
    // Scatter: single runs of one row, each after a churn.
    let scatter = state().scatter;
    if let Some((vi, k)) = scatter {
        for i in 0..k {
            if STOP.load(Ordering::SeqCst) {
                break;
            }
            churn();
            do_run(vi, -100 - i as i32);
        }
    }
    // Warm-up, table order. Every warm run starts after a churn.
    let t_warm = now_s();
    for vi in 0..n {
        if STOP.load(Ordering::SeqCst) {
            break;
        }
        churn();
        do_run(vi, -1);
    }
    let round_s = now_s() - t_warm;
    let mut est = round_s;
    for round in 0..reps {
        if STOP.load(Ordering::SeqCst) {
            state().passes[pi].note += &format!(" watchdog after {round} rounds");
            break;
        }
        if now_s() + est > soft {
            state().passes[pi].note += &format!(" soft budget: stopped after {round} of {reps} rounds (round ~{est:.1}s)");
            note(format!("P{} soft budget stop at round {round}", pi + 1));
            break;
        }
        let t = now_s();
        for k in 0..n {
            if STOP.load(Ordering::SeqCst) {
                break;
            }
            churn();
            do_run((k + round) % n, round as i32);
        }
        est = now_s() - t;
        if !STOP.load(Ordering::SeqCst) {
            state().passes[pi].rounds_done = round + 1;
        }
    }
    let st = state();
    let ok: Vec<f64> = st.runs.iter().filter(|r| r.pass == pi && timed(r)).map(|r| r.p.wall_us as f64 / 1e6).collect();
    if ok.is_empty() { None } else { Some(ok.iter().sum::<f64>() / ok.len() as f64) }
}

/// A run whose wall time counts: it finished normally (a different answer
/// is flagged, but its time is still a time).
fn timed(r: &Run) -> bool {
    r.p.status == "ok" || r.p.status == "DIFF"
}

// ---------------------------------------------------------------- fake engine

fn fake(mode: &str) -> ! {
    eprintln!("[engine] parse      wall     1.0 ms  cpu     1.0 ms  (at     1.0 ms)");
    match mode {
        "crash" => std::process::abort(),
        "fail" => std::process::exit(3),
        _ => {
            eprintln!("[engine] self-check user_daily sessions 1 revenue 2 rows 3; campaign_daily sessions 1 revenue 2 rows 3; OK");
            std::process::exit(0)
        }
    }
}

// ---------------------------------------------------------------- main

fn main() {
    if let Ok(m) = std::env::var("BENCH_FAKE") {
        fake(&m);
    }
    START.get_or_init(Instant::now);
    let soft = knob_f64("SOFT_S", SOFT_S);
    let hard = knob_f64("HARD_S", HARD_S).min(145.0);
    eprintln!("BENCHSTART soft_s={soft} hard_s={hard}");
    std::thread::spawn(move || {
        while now_s() < hard {
            std::thread::sleep(Duration::from_millis(200));
        }
        STOP.store(true, Ordering::SeqCst);
        let pid = CHILD.load(Ordering::SeqCst);
        eprintln!("BENCHWATCHDOG at {:.1}s, killing pid {pid}", now_s());
        if pid > 0 {
            unsafe { kill(pid, 9) };
        }
        // Give the main thread a moment to finish on its own.
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(50));
        }
        finish("watchdog");
    });

    let data_in = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out_dir = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let env = environment();
    for (k, v) in &env {
        eprintln!("ENV {k}={v}");
    }
    {
        let mut st = state();
        st.env = env;
        st.out_dir = out_dir.clone();
        st.day = window_day(&data_in);
    }
    let table = std::env::var("BENCH_TABLE").unwrap_or_else(|_| TABLE.into());
    let variants = match load_table(&table) {
        Ok(v) => v,
        Err(e) => {
            note(format!("table: {e}"));
            finish("table");
        }
    };
    for v in &variants {
        eprintln!("VARIANT {} {}", v.name, v.spec);
    }
    state().variants = variants.clone();
    let first = load_first(&table, &variants);
    eprintln!("FIRST {}", first.iter().map(|&i| variants[i].name.as_str()).collect::<Vec<_>>().join(","));
    state().first = first;
    state().scatter = load_scatter(&table, &variants);

    // 1. The real answer, by the first variant (scorer's dev check).
    let (p, phases, check) = {
        let _ = fs::create_dir_all(&out_dir);
        let mut cmd = Command::new(&variants[0].bin);
        base_env(&mut cmd);
        cmd.env("DATA_DIR", &data_in).env("OUTPUT_DIR", &out_dir).env("ENGINE_TIMING", "1");
        for (k, val) in &variants[0].env {
            cmd.env(k, val);
        }
        let p = run_proc(cmd, &scratch().join("dev.log"));
        let (ph, ck) = parse_engine(&p.stderr);
        (p, ph, ck)
    };
    for l in p.stderr.lines().filter(|l| l.starts_with("[engine] ") && !l.starts_with("[engine]   ")) {
        eprintln!("DEVLOG {l}");
    }
    if p.status != "ok" {
        eprintln!("DEVFAIL {} tail: {}", p.status, tail(&p.stderr, 5));
    }
    state().dev = Some(Run { pass: 0, variant: 0, rep: -2, p, phases, check, out: "-".into() });
    // Provisional rows, in case the driver dies before the end.
    write_rows(&["bench started; no results yet (driver died before the report)".to_string()]);

    // 2. First pass.
    let ev = knob_u64("EVENTS", EVENTS);
    let cl = knob_u64("CLICKS", CLICKS);
    let t1 = now_s();
    let reps = knob_u64("REPS", REPS as u64) as usize;
    let mean = pass(0, ev, cl, reps, soft);
    let gen_failed = state().passes[0].genp.as_ref().is_some_and(|g| g.status != "ok");
    if gen_failed && !STOP.load(Ordering::SeqCst) {
        // For example a full /scratch: one retry at a quarter of the size.
        note(format!("P1 generator failed; retry at {} events", ev / 4));
        let _ = fs::remove_dir_all(scratch().join("data"));
        pass(1, ev / 4, cl / 4, reps, soft);
        finish("done (reduced size)");
    }
    let p1 = now_s() - t1;

    // 3. Optional second pass, if its estimate fits.
    let ev2 = knob_u64("EVENTS2", EVENTS2);
    let cl2 = knob_u64("CLICKS2", CLICKS2);
    let reps2 = knob_u64("REPS2", REPS2 as u64) as usize;
    if ev2 > 0 && !STOP.load(Ordering::SeqCst) {
        if let Some(mean) = mean {
            let ratio = ev2 as f64 / ev.max(1) as f64;
            let (gen_s, sync_s) = {
                let st = state();
                let g = st.passes[0].genp.as_ref().map_or(0.0, |g| g.wall_us as f64 / 1e6);
                (g, st.passes[0].sync_ms / 1e3)
            };
            let n = variants.len() as f64;
            let est = (gen_s + sync_s) * ratio * 1.3 + mean * ratio * n * (1 + reps2) as f64;
            if now_s() + est <= soft {
                pass(1, ev2, cl2, reps2, soft);
            } else {
                note(format!(
                    "P2 {}M skipped: estimate {est:.0}s, now {:.0}s, soft {soft:.0}s (pass 1 took {p1:.0}s)",
                    ev2 / 1_000_000,
                    now_s()
                ));
            }
        }
    }
    finish("done");
}
