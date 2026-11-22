//! End-to-end differential tests: the fast engine against the reference
//! pipeline on the kit fixtures (when present) and on small random
//! datasets built from deliberately awkward values.

use std::path::{Path, PathBuf};

use crate::common::*;
use crate::engine::{self, Config};
use crate::naive;

fn cfg(threads: usize, chunk: usize, bits: u32) -> Config {
    Config {
        threads,
        timing: false,
        collect: true,
        chunk_bytes: chunk,
        p1_bits: Some(bits),
        p2_bits: Some(bits),
        onepass: true,
        dense: false,
        dense_slots: None,
        fused: true,
    }
}

fn compare(data: &Path, config: &Config) -> [i64; 5] {
    let lk = load_lookups(data);
    let want = naive::run(data, &lk);
    // All shapes of the engine: one-pass, three-phase and dense.
    let other = Config { onepass: !config.onepass, ..*config };
    let dense = Config { dense: true, ..*config };
    // And the two-stage reader for events and clicks.
    let staged = Config { fused: !config.fused, onepass: false, ..*config };
    for config in [config, &other, &dense, &staged] {
        check_shape(data, config, &lk, &want);
    }
    [
        want.user.len() as i64,
        want.user.iter().filter(|r| r.top.is_some()).count() as i64,
        want.user.iter().map(|r| r.sessions).sum(),
        want.camp.iter().filter(|r| r.campaign != b"organic").map(|r| r.sessions).sum(),
        want.camp.iter().map(|r| r.assisted).sum(),
    ]
}

/// Run one shape and compare it with the reference; true when the dense
/// shape ran to the end.
fn check_shape(data: &Path, config: &Config, lk: &Lookups, want: &Tables) -> bool {
    let mut got = engine::run(data, None, lk, config, std::time::Instant::now()).expect("fast engine declined");
    got.sort();
    // The dense shape gives up only when its table fills or a click record
    // cannot hold its fields; the tests that force that set a tiny table.
    if config.dense && config.dense_slots.is_none() {
        assert!(got.dense, "{}: the dense shape gave up", data.display());
    }
    let data = Path::new(data);
    let shape = if config.dense {
        "dense"
    } else if config.onepass {
        "one-pass"
    } else {
        "three-phase"
    };
    let data = &data.join(shape);
    if got.user != want.user {
        for (a, b) in got.user.iter().zip(&want.user) {
            if a != b {
                panic!("{}: user_daily differs\n fast {a:?}\n want {b:?}", data.display());
            }
        }
        panic!("{}: user_daily rows {} vs {}", data.display(), got.user.len(), want.user.len());
    }
    if got.camp != want.camp {
        for (a, b) in got.camp.iter().zip(&want.camp) {
            if a != b {
                panic!("{}: campaign_daily differs\n fast {a:?}\n want {b:?}", data.display());
            }
        }
        panic!("{}: campaign_daily rows {} vs {}", data.display(), got.camp.len(), want.camp.len());
    }
    got.dense
}

fn fixtures_dir() -> Option<PathBuf> {
    let dir = std::env::var("SESSIONS_FIXTURES").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from("/Users/yamanakbas/Developer/buildathon-2026/tracks/shopping-sessions/kit/starter-duckdb/starter-duckdb/fixtures")
    });
    dir.is_dir().then_some(dir)
}

#[test]
fn fixtures_match_reference() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("fixtures not found, skipped");
        return;
    };
    let mut cases = 0;
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let input = entry.path().join("input");
        if input.is_dir() {
            compare(&input, &cfg(1, 4 << 20, 1));
            compare(&input, &cfg(3, 64, 3));
            cases += 1;
        }
    }
    assert!(cases >= 36);
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, one_in: u64) -> bool {
        self.below(one_in) == 0
    }
    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len() as u64) as usize]
    }
}

fn utc(ms: i64) -> String {
    let (y, m, d) = civil_from_days(ms.div_euclid(DAY_MS));
    let s = ms.rem_euclid(DAY_MS) / 1000;
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// A timestamp field for `ms`: one of the three valid forms, or a bad one.
fn ts_field(r: &mut Rng, ms: i64) -> String {
    if r.chance(25) {
        return r
            .pick(&[
                "", "2026-10-03 10:00:00Z", "2026-10-03T10:00:00z", "2026-10-03t10:00:00Z", "2026-10-03T10:00:00",
                "1791071890000.0", "1.79e12", "-5", "2026-13-03T10:00:00Z", "2026-02-30T10:00:00Z",
                "2026-10-03T24:00:00Z", "2026-10-03T10:00:60Z", "2026-10-03T10:00:00+24:00", "2026-10-03T10:00:00+05:60",
                "2026-10-03T10:00:00+0200", "86400000", "123456789012345678901234", "2026-10-03T10:00:00.000Z",
                "\u{0661}\u{0667}\u{0669}", "2026-10-03T10:00:00Z\r", "2027-02-29T10:00:00Z",
            ])
            .to_string();
    }
    if ms % 1000 != 0 || r.chance(3) {
        return if r.chance(6) { format!("000{ms}") } else { format!("{ms}") };
    }
    if r.chance(4) {
        let offset = r.below(48) as i64 * 30 - 720 + if r.chance(5) { 719 } else { 0 };
        let sign = if offset < 0 { '-' } else { '+' };
        return format!("{}{sign}{:02}:{:02}", utc(ms + offset * 60_000), offset.abs() / 60, offset.abs() % 60);
    }
    format!("{}Z", utc(ms))
}

fn emit(r: &mut Rng, fields: &[String], out: &mut Vec<u8>, allow_cr: bool) {
    let mut line = String::new();
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        if r.chance(5) {
            line.push('"');
            line.push_str(f);
            line.push('"');
        } else {
            line.push_str(f);
        }
    }
    if r.chance(40) {
        match r.below(8) {
            0 => {
                let cut = line.rfind(',').unwrap_or(0);
                line.truncate(cut);
            }
            1 => line.push_str(",x"),
            2 => line.insert(0, '"'),
            3 => line.push('"'),
            4 => line = String::new(),
            5 => {
                let at = r.below(line.len() as u64 + 1) as usize;
                if line.is_char_boundary(at) {
                    line.insert(at, '"');
                }
            }
            6 if allow_cr => line.push('\r'),
            6 => line.push_str(","),
            _ => line = line.replace(',', "\",\""),
        }
    }
    out.extend_from_slice(line.as_bytes());
    out.push(b'\n');
}

fn write_parts(dir: &Path, header: &str, bodies: Vec<Vec<u8>>, drop_last_newline: bool) {
    std::fs::create_dir_all(dir).unwrap();
    for (i, mut body) in bodies.into_iter().enumerate() {
        let mut bytes = format!("{header}\n").into_bytes();
        if drop_last_newline && i == 0 && body.last() == Some(&b'\n') {
            body.pop();
        }
        bytes.extend_from_slice(&body);
        std::fs::write(dir.join(format!("part-{i:05}.csv")), bytes).unwrap();
    }
}

fn generate(dir: &Path, seed: u64, events: usize) {
    let mut r = Rng(seed);
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    // Windows of odd shapes: not at midnight, short, long, around a leap day.
    let ws = match seed % 4 {
        0 => days_from_civil(2026, 10, 1) * DAY_MS,
        1 => days_from_civil(2028, 2, 20) * DAY_MS + 3_600_000 * 5,
        2 => days_from_civil(2027, 12, 28) * DAY_MS,
        _ => days_from_civil(2025, 3, 1) * DAY_MS + 1000,
    };
    let days = [14, 3, 1, 30][(seed / 4 % 4) as usize];
    let we = ws + days * DAY_MS;
    std::fs::write(
        dir.join("manifest.json"),
        format!("{{\n  \"window_start\": \"{}Z\",\n  \"window_end\": \"{}Z\"\n}}\n", utc(ws), utc(we)),
    )
    .unwrap();
    let cats = ["shoes", "books", "unknowm", "zzz", "apparel", "a"];
    let mut products = String::from("product_id,category\n");
    for p in 1..=40 {
        products.push_str(&format!("p{p},{}\n", cats[r.below(cats.len() as u64) as usize]));
    }
    products.push_str("P7,zzz\np007,books\nwidget,shoes\n p3,a\n");
    std::fs::write(dir.join("products.csv"), products).unwrap();
    let mut campaigns = String::from("campaign_id,channel\n");
    for c in 1..=12 {
        campaigns.push_str(&format!("c_{c},{}\n", ["search", "social", "email"][c % 3]));
    }
    campaigns.push_str("promo,email\nc_007,social\n");
    std::fs::write(dir.join("campaigns.csv"), campaigns).unwrap();

    let odd_users = ["U1", " u1", "u1 ", "A\tB", "user_id_a", "averyveryverylonguserid", "u\u{00e9}", "u1", "u2"];
    let user = |r: &mut Rng| -> String {
        if r.chance(60) {
            return String::new();
        }
        if r.chance(12) {
            return r.pick(&odd_users).to_string();
        }
        format!("u{}", r.below(events as u64 / 6 + 3))
    };
    let nparts = 1 + r.below(4) as usize;
    let mut event_parts = vec![Vec::new(); nparts + 1];
    let mut click_parts = vec![Vec::new(); nparts + 1];
    let mut link_parts = vec![Vec::new(); 3];
    let span = (we - ws) as u64;
    let time = |r: &mut Rng| -> i64 {
        let slot = ws + (r.below(span / 600_000 + 1) * 600_000) as i64;
        match r.below(12) {
            0 => slot + 1_800_000,
            1 => slot + 1_800_001,
            2 => slot + r.below(999) as i64,
            3 => ws - 1 + r.below(3) as i64,
            4 => we - 2 + r.below(3) as i64,
            _ => slot + (r.below(400) * 1000) as i64,
        }
    };
    for n in 0..events {
        let ts = time(&mut r);
        let id = match r.below(200) {
            0 => String::new(),
            1 => format!("E{n}"),
            2 => format!("e{n} "),
            3 => format!("event-with-a-long-id-{n}"),
            _ => format!("e{n}"),
        };
        let kind = match r.below(40) {
            0 => r.pick(&["Purchase", "PURCHASE", "View", "cart ", "click", "", "views", " view"]).to_string(),
            1..=8 => "purchase".to_string(),
            9..=14 => "cart".to_string(),
            _ => "view".to_string(),
        };
        let product = match r.below(30) {
            0 => String::new(),
            1 => r.pick(&["P7", "p007", "widget", " p3", "p99999", "p0", "p12345678", "px"]).to_string(),
            _ => format!("p{}", 1 + r.below(45)),
        };
        let price = match r.below(12) {
            0 => r.pick(&["19.99", "-5", "+500", "0", "", "10000001", "250000000", "1e3", "00", "0250", "10000000", "1"]).to_string(),
            1 => String::new(),
            _ => format!("{}", 100 + r.below(5000)),
        };
        let ingest = match r.below(30) {
            0 => ts - ts.rem_euclid(1000) + DAY_MS,
            1 => ts - ts.rem_euclid(1000) + DAY_MS + 1000,
            2 => ts - ts.rem_euclid(1000) - 5000,
            3 => ts + 3 * DAY_MS,
            _ => ts - ts.rem_euclid(1000) + (r.below(600) * 1000) as i64 + 1000,
        };
        let copies = if r.chance(12) { 2 + r.below(3) } else { 1 };
        let who = user(&mut r);
        // Copies with an equal ingest_ts are identical (rules): the base
        // row is built once, and a copy that differs gets its own, strictly
        // later ingest_ts.
        let base = vec![
            id.clone(),
            who.clone(),
            ts_field(&mut r, ts),
            format!("{}Z", utc(ingest - ingest.rem_euclid(1000))),
            kind.clone(),
            product.clone(),
            price.clone(),
        ];
        for copy in 0..copies {
            let mut fields = base.clone();
            if copy > 0 && r.chance(3) {
                fields[3] = format!("{}Z", utc(ingest - ingest.rem_euclid(1000) + (copy * 1000) as i64));
                match r.below(4) {
                    0 => fields[1] = user(&mut r),
                    1 => {
                        let other = time(&mut r);
                        fields[2] = ts_field(&mut r, other);
                    }
                    2 => fields[4] = "purchase".into(),
                    _ => {}
                }
            } else if copy > 0 && r.chance(8) {
                // An earlier copy that is invalid: it must kill the event.
                fields[3] = format!("{}Z", utc(ingest - ingest.rem_euclid(1000) - (copy * 1000) as i64));
                fields[2] = "2026-10-03 10:00:00".into();
            }
            let part = r.below(nparts as u64) as usize;
            emit(&mut r, &fields, &mut event_parts[part], copies == 1);
        }
    }
    for n in 0..events / 3 {
        let ts = match r.below(10) {
            0 => ws - CLICK_REACH_MS - 1 + r.below(3) as i64,
            1 => time(&mut r) - CLICK_REACH_MS,
            2 => time(&mut r) - CLICK_REACH_MS - 1,
            3 => time(&mut r) + 1,
            _ => time(&mut r) - (r.below(9 * 86_400) * 1000) as i64,
        };
        let id = match r.below(100) {
            0 => String::new(),
            1 => format!("K{n}"),
            2 => format!("click-with-a-long-id-{n}"),
            _ => format!("k{n}"),
        };
        let camp = match r.below(25) {
            0 => r.pick(&["C_1", "c_999", "c_007", "promo", "", "c_", "c_01", " c_1", "organic"]).to_string(),
            _ => format!("c_{}", 1 + r.below(12)),
        };
        let who = user(&mut r);
        let stamp = ts_field(&mut r, ts);
        let fields = vec![id, who.clone(), camp, stamp.clone()];
        for _ in 0..if r.chance(15) { 2 } else { 1 } {
            let part = r.below(nparts as u64) as usize;
            emit(&mut r, &fields, &mut click_parts[part], false);
        }
        // A second click at the same instant with another id: the tie rule.
        if r.chance(6) {
            let twin_id = if r.chance(4) { format!("click-twin-long-id-{n}") } else { format!("j{n}") };
            let twin = vec![twin_id, who, format!("c_{}", 1 + r.below(12)), stamp];
            emit(&mut r, &twin, &mut click_parts[0], true);
        }
    }
    for _ in 0..events / 8 {
        let fields = vec![user(&mut r), user(&mut r)];
        let part = r.below(2) as usize;
        emit(&mut r, &fields, &mut link_parts[part], true);
    }
    if seed % 5 == 0 {
        link_parts[0].extend_from_slice(b"user_id_a,user_id_b\n");
    }
    write_parts(&dir.join("events"), "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents", event_parts, seed % 3 == 0);
    write_parts(&dir.join("clicks"), "click_id,user_id,campaign_id,ts", click_parts, seed % 3 == 1);
    write_parts(&dir.join("links"), "user_id_a,user_id_b", link_parts, seed % 3 == 2);
}

/// Extra part files for a generated dataset whose lines are far longer than
/// a scan window (64 KB): a long field in every position, valid and not,
/// quoted and not, copies of one event among them, and malformed long lines.
fn add_long_lines(dir: &Path, seed: u64) {
    let mut r = Rng(seed ^ 0x10C0);
    let lk = load_lookups(dir);
    let t = lk.window_start - lk.window_start.rem_euclid(1000) + 3_600_000;
    let f1 = |ms: i64| format!("{}Z", utc(ms));
    let mut long = |byte: char| -> String { std::iter::repeat_n(byte, 66_000 + r.below(150_000) as usize).collect() };
    let row = |fields: &[&str]| -> String { fields.join(",") + "\n" };
    let quoted = |s: &str| format!("\"{s}\"");
    let (id_a, id_b, user_y, name_a) = (long('x'), long('w'), long('y'), long('A'));
    let zeros = long('0');
    let mut events = String::from("event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
    // One event with a long id in two copies: the earlier one is kept.
    events += &row(&[&id_a, "u1", &f1(t), &f1(t + 60_000), "view", "p1", ""]);
    events += &row(&[&quoted(&id_a), "u2", &f1(t), &f1(t + 120_000), "view", "p1", ""]);
    // Another whose earliest copy is invalid: the event is dropped.
    events += &row(&[&id_b, "u1", &f1(t + 1000), &f1(t + 60_000), "view", "p1", ""]);
    events += &row(&[&id_b, "u1", "2026-10-03 10:00:00Z", &quoted(&f1(t + 30_000)), "view", "p1", ""]);
    events += &row(&["L1", &user_y, &f1(t), &f1(t + 5000), "purchase", "p2", "250"]);
    events += &row(&["L2", "u1", &format!("{zeros}{}", t + 7), &f1(t + 5000), "cart", "p3", ""]);
    events += &row(&["L3", "u2", &f1(t), &format!("{zeros}{}", t + 9000), "view", "p1", ""]);
    events += &row(&["L3b", "u2", &f1(t), &format!("{zeros}{}", t + 9000 + 2 * DAY_MS), "view", "p1", ""]);
    events += &row(&["L4", "u1", &f1(t), &f1(t + 5000), &long('v'), "p1", ""]);
    events += &row(&["L5", "u1", &f1(t + 2000), &f1(t + 5000), "purchase", &quoted(&long('p')), "300"]);
    events += &row(&["L6", "u2", &f1(t + 3000), &f1(t + 5000), "purchase", "p4", &format!("{zeros}0250")]);
    events += &row(&["L7", "u2", &f1(t + 4000), &f1(t + 5000), "purchase", "p4", &long('9')]);
    events += &row(&["L8", "u2", &f1(t + 4000), &long('2'), "view", "p4", ""]);
    events += &row(&["L9", "u2", &long('T'), &f1(t + 5000), "view", "p4", ""]);
    // A long copy that takes an event of the generated files over.
    events += &row(&["e5", &user_y, &f1(t + 60_000), &f1(t - 10 * DAY_MS), "view", "p1", ""]);
    // Malformed: a field too many, one too few, a quote inside, commas only.
    events += &row(&["M1", "u1", &f1(t), &f1(t), "view", "p1", "", &long('z')]);
    events += &row(&["M2", "u1", &f1(t), &f1(t), "view", &long('z')]);
    events += &row(&["M3", "u1", &f1(t), &f1(t), "view", &format!("{}\"{}", long('q'), long('q')), ""]);
    events += &row(&["M4", "u1", &f1(t), &f1(t), "view", &format!("\"{}", long('q')), ""]);
    events += &long(',');
    events += "\n";
    events += &row(&["N1", "u1", &f1(t + 300_000), &f1(t + 300_000), "view", "p1", ""]);
    events += &row(&["N2", &name_a, &f1(t + 9 * 3_600_000), &f1(t + 9 * 3_600_000), "purchase", "p5", "777"]);
    if seed % 2 == 0 {
        // The file ends inside a long line.
        events += &row(&["N3", "u2", &f1(t + 400_000), &f1(t + 400_000), "view", &long('p'), ""]);
        events.pop();
    }
    std::fs::write(dir.join("events/part-90000.csv"), events).unwrap();

    let click_k = long('k');
    let mut clicks = String::from("click_id,user_id,campaign_id,ts\n");
    // Two clicks at one instant: the long id is the smaller one and wins.
    clicks += &row(&[&click_k, "u1", "c_1", &f1(t - 3_600_000)]);
    clicks += &row(&["kzz", "u1", "c_2", &f1(t - 3_600_000)]);
    clicks += &row(&["K2", &quoted(&user_y), "c_3", &f1(t - 7_200_000)]);
    clicks += &row(&["K3", "u1", &long('c'), &f1(t - 60_000)]);
    clicks += &row(&["K4", "u2", "c_4", &format!("{zeros}{}", t - 1_800_000)]);
    clicks += &row(&["K5", "u2", "c_5", &long('9')]);
    clicks += &row(&["K6", "u2", "c_5", &f1(t - 1000), &long('z')]);
    clicks += &row(&["K7", &name_a, "c_6", &f1(t + 8 * 3_600_000)]);
    clicks += &long(',');
    clicks += "\n";
    clicks += &row(&["K8", "u1", "c_7", &f1(t + 200_000)]);
    if seed % 2 == 1 {
        clicks += &row(&[&long('j'), "u2", "c_8", &f1(t + 100_000)]);
        clicks.pop();
    }
    std::fs::write(dir.join("clicks/part-90000.csv"), clicks).unwrap();

    let mut links = String::from("user_id_a,user_id_b\n");
    links += &row(&[&user_y, "u2"]);
    links += &row(&["u1", &quoted(&name_a)]);
    links += &row(&[&long('m'), ""]);
    links += &row(&[&long('m'), "u1", "u2"]);
    links += &row(&[&format!("{}\"", long('m')), "u1"]);
    links += &row(&["u3", "u4"]);
    if seed % 3 == 0 {
        links += &row(&[&long('B'), &long('C')]);
        links.pop();
    }
    std::fs::write(dir.join("links/part-90000.csv"), links).unwrap();
}

/// Lines longer than a scan window take the slice path. Its rows must be
/// the reference's rows, whatever the chunk size.
#[test]
fn long_lines_match_reference() {
    let root = std::env::temp_dir().join(format!("sessions-long-{}", std::process::id()));
    let mut long_names = 0;
    for seed in 0..6u64 {
        let dir = root.join(format!("s{seed}"));
        generate(&dir, seed, 300);
        add_long_lines(&dir, seed);
        let stats = compare(&dir, &cfg(3, 5000, 2));
        assert_eq!(compare(&dir, &cfg(1, 4 << 20, 1)), stats);
        assert_eq!(compare(&dir, &cfg(4, 256, 4)), stats);
        // Rows named by a long ID (when no shorter ID of the person sorts
        // before it) show that the long lines reached the output.
        let lk = load_lookups(&dir);
        long_names += naive::run(&dir, &lk).user.iter().filter(|row| row.user_id.len() > 60_000).count();
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(long_names >= 6, "only {long_names} rows with a long name");
    let _ = std::fs::remove_dir_all(&root);
}

/// A dataset whose events file has one line with a `hole`-byte product_id
/// (all NUL bytes), written as a sparse file. `false` when the file system
/// stored the hole for real.
fn giant_line_dataset(dir: &Path, hole: u64) -> bool {
    use std::io::{Seek, SeekFrom, Write};
    use std::os::unix::fs::MetadataExt;
    let _ = std::fs::remove_dir_all(dir);
    for sub in ["events", "clicks", "links"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    std::fs::write(
        dir.join("manifest.json"),
        "{\"window_start\": \"2026-10-01T00:00:00Z\", \"window_end\": \"2026-10-15T00:00:00Z\"}\n",
    )
    .unwrap();
    std::fs::write(dir.join("products.csv"), "product_id,category\np1,shoes\np2,books\n").unwrap();
    std::fs::write(dir.join("campaigns.csv"), "campaign_id,channel\nc_1,search\n").unwrap();
    std::fs::write(dir.join("clicks/part-00000.csv"), "click_id,user_id,campaign_id,ts\nk1,u1,c_1,2026-10-03T09:00:00Z\n").unwrap();
    std::fs::write(dir.join("links/part-00000.csv"), "user_id_a,user_id_b\nu4,u5\n").unwrap();
    let path = dir.join("events/part-00000.csv");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(b"event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n").unwrap();
    file.write_all(b"e1,u1,2026-10-03T10:00:00Z,2026-10-03T10:00:05Z,purchase,p1,500\n").unwrap();
    // The giant line is the kept copy of g1: a later, ordinary copy names
    // another user, so losing the giant line moves the event to `u3`.
    file.write_all(b"g1,u1,2026-10-05T10:01:00Z,2026-10-05T10:01:05Z,view,").unwrap();
    file.seek(SeekFrom::Current(hole as i64)).unwrap();
    file.write_all(b",\n").unwrap();
    file.write_all(b"g1,u3,2026-10-05T10:01:00Z,2026-10-05T10:01:09Z,view,p1,\n").unwrap();
    file.write_all(b"e2,u2,2026-10-04T10:00:00Z,2026-10-04T10:00:05Z,purchase,p2,700\n").unwrap();
    drop(file);
    let meta = std::fs::metadata(&path).unwrap();
    assert!(meta.len() > hole);
    meta.blocks() * 512 < (64 << 20)
}

fn giant_line_case(hole: u64) {
    let dir = std::env::temp_dir().join(format!("sessions-giant-{}-{}", std::process::id(), hole >> 20));
    if !giant_line_dataset(&dir, hole) {
        eprintln!("no sparse files here, skipped");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let stats = compare(&dir, &cfg(4, 4 << 20, 2));
    // u1 has two days (the purchase and the giant view), u2 one; no u3.
    assert_eq!(stats[0], 3);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The rules set no maximum line length. One line of over 1 GiB (which the
/// engine used to skip) is an ordinary event. The input is a sparse file,
/// so it costs no disk; the reference still reads it into memory.
#[test]
fn line_over_one_gib_matches_reference() {
    giant_line_case((1 << 30) + 4097);
}

/// The same past 4 GiB, where a 32-bit offset would wrap. Costs 5 GiB of
/// memory for the reference, so it only runs on request.
#[test]
#[ignore]
fn line_over_four_gib_matches_reference() {
    giant_line_case((5 << 30) + 4097);
}

#[test]
fn random_datasets_match_reference() {
    let root = std::env::temp_dir().join(format!("sessions-fuzz-{}", std::process::id()));
    let mut totals = [0i64; 5];
    for seed in 0..120u64 {
        let dir = root.join(format!("s{seed}"));
        let events = [40, 300, 3000, 12000][(seed % 4) as usize];
        generate(&dir, seed, events);
        let (threads, chunk, bits) = match seed % 3 {
            0 => (1, 4 << 20, 1),
            1 => (4, 256, 4),
            _ => (3, 5000, 2),
        };
        let stats = compare(&dir, &cfg(threads, chunk, bits));
        for k in 0..5 {
            totals[k] += stats[k];
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    // The datasets must exercise the pipeline: rows, purchases, credited
    // sessions and assists all present in bulk.
    eprintln!("fuzz totals [rows, rows with top, sessions, credited, assists] = {totals:?}");
    assert!(totals[0] > 20_000 && totals[1] > 2_000 && totals[3] > 2_000 && totals[4] > totals[3]);
    let _ = std::fs::remove_dir_all(&root);
}

/// Events are grouped by time bin (2^20 ms) and never sorted, so this packs
/// the times of a few busy people around bin edges: one millisecond either
/// side of an edge, equal times, and gaps of exactly 30:00.000 and
/// 30:00.001 that cross one or two edges, with purchases all over.
#[test]
fn bin_edges_match_reference() {
    let root = std::env::temp_dir().join(format!("sessions-bins-{}", std::process::id()));
    let mut sessions = 0;
    for seed in 0..9u64 {
        let mut r = Rng(seed + 1000);
        let dir = root.join(format!("b{seed}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ws = days_from_civil(2026, 10, 1) * DAY_MS + [0, 777_000, 3_600_000][(seed % 3) as usize];
        let days = [14, 2, 40][(seed / 3 % 3) as usize];
        let we = ws + days * DAY_MS;
        std::fs::write(
            dir.join("manifest.json"),
            format!("{{\"window_start\": \"{}Z\", \"window_end\": \"{}Z\"}}\n", utc(ws), utc(we)),
        )
        .unwrap();
        std::fs::write(dir.join("products.csv"), "product_id,category\np1,shoes\np2,books\np3,a\np4,shoes\n").unwrap();
        std::fs::write(dir.join("campaigns.csv"), "campaign_id,channel\nc_1,search\nc_2,social\nc_3,email\n").unwrap();
        let bin = 1i64 << 20;
        let mut times: Vec<i64> = Vec::new();
        let mut events = String::from("event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
        let mut n = 0;
        for _ in 0..500 {
            let edge = ws + r.below((we - ws) as u64 / bin as u64 + 1) as i64 * bin;
            let who = r.below(4);
            let mut t = edge + r.below(5) as i64 - 2;
            for _ in 0..1 + r.below(6) {
                if (ws..we).contains(&t) {
                    let kind = if r.chance(3) { "purchase" } else { "view" };
                    let ingest = t - t.rem_euclid(1000) + 1000;
                    events.push_str(&format!("b{n},u{who},{t},{}Z,{kind},p{},{}\n", utc(ingest), 1 + r.below(5), 100 + r.below(900)));
                    times.push(t);
                    n += 1;
                }
                t += match r.below(8) {
                    0 => 0,
                    1 => SESSION_GAP_MS,
                    2 => SESSION_GAP_MS + 1,
                    3 => bin - 1,
                    4 => bin,
                    5 => SESSION_GAP_MS - 1,
                    6 => 1,
                    _ => r.below(2 * bin as u64) as i64,
                };
            }
        }
        std::fs::create_dir_all(dir.join("events")).unwrap();
        std::fs::write(dir.join("events/part-0.csv"), events).unwrap();
        let mut clicks = String::from("click_id,user_id,campaign_id,ts\n");
        for k in 0..300 {
            let at = times[r.below(times.len() as u64) as usize];
            let t = at + [0, 1, -1, -CLICK_REACH_MS, -CLICK_REACH_MS - 1, -3_600_000][r.below(6) as usize];
            clicks.push_str(&format!("k{k},u{},c_{},{t}\n", r.below(4), 1 + r.below(3)));
        }
        std::fs::create_dir_all(dir.join("clicks")).unwrap();
        std::fs::write(dir.join("clicks/part-0.csv"), clicks).unwrap();
        std::fs::create_dir_all(dir.join("links")).unwrap();
        let links = if seed % 2 == 0 { "user_id_a,user_id_b\nu0,u1\n" } else { "user_id_a,user_id_b\n" };
        std::fs::write(dir.join("links/part-0.csv"), links).unwrap();
        sessions += compare(&dir, &cfg(2, 4096, 1))[2];
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(sessions > 2_000, "only {sessions} sessions");
    let _ = std::fs::remove_dir_all(&root);
}

/// People with 1 to 40 sessions, so both click paths run and the one for few
/// sessions runs with one to four blocks. Clicks sit on the edges of reach
/// of a session (at its start, 1 ms after, exactly 7 days before, 1 ms more
/// or less), come in twins at one instant with the same or another
/// campaign, and as exact copies. Odd seeds use a window too long for the
/// 32-bit path.
#[test]
fn click_reach_edges_match_reference() {
    let root = std::env::temp_dir().join(format!("sessions-reach-{}", std::process::id()));
    let (mut credited, mut assists) = (0, 0);
    for seed in 0..8u64 {
        let mut r = Rng(seed + 2000);
        let dir = root.join(format!("c{seed}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ws = days_from_civil(2027, 3, 1) * DAY_MS + [0, 5_000][(seed / 2 % 2) as usize];
        let days = if seed % 2 == 0 { 14 } else { 20 };
        let we = ws + days * DAY_MS;
        std::fs::write(
            dir.join("manifest.json"),
            format!("{{\"window_start\": \"{}Z\", \"window_end\": \"{}Z\"}}\n", utc(ws), utc(we)),
        )
        .unwrap();
        std::fs::write(dir.join("products.csv"), "product_id,category\np1,shoes\np2,books\n").unwrap();
        std::fs::write(dir.join("campaigns.csv"), "campaign_id,channel\nc_1,search\nc_2,social\nc_3,email\nc_4,email\nc_5,search\n").unwrap();
        let mut events = String::from("event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
        let mut clicks = String::from("click_id,user_id,campaign_id,ts\n");
        let (mut n, mut k) = (0, 0);
        for who in 0..40u64 {
            let count = who as i64 + 1;
            for j in 0..count {
                // Sessions hours apart, a few of them packed on one day.
                let start = ws + (we - ws - 3_600_000) * j / count + r.below(1000) as i64;
                for step in 0..1 + r.below(3) as i64 {
                    let t = start + step * 5_000;
                    let kind = if r.chance(4) { "purchase" } else { "view" };
                    let ingest = t - t.rem_euclid(1000) + 1000;
                    events.push_str(&format!("r{n},w{who},{t},{}Z,{kind},p{},{}\n", utc(ingest), 1 + r.below(2), 100 + r.below(900)));
                    n += 1;
                }
                for _ in 0..r.below(4) {
                    let t = start + [0, 1, -1, -CLICK_REACH_MS, -CLICK_REACH_MS - 1, -CLICK_REACH_MS + 1, -3 * 3_600_000, -DAY_MS][r.below(8) as usize];
                    let line = format!("q{k},w{who},c_{},{t}\n", 1 + r.below(5));
                    clicks.push_str(&line);
                    k += 1;
                    match r.below(6) {
                        // An exact copy, a twin of the same campaign, and
                        // twins of any campaign with a smaller or larger id.
                        0 => clicks.push_str(&line),
                        1 => clicks.push_str(&format!("q{k}x,w{who},{},{t}\n", line.split(',').nth(2).unwrap())),
                        2 => clicks.push_str(&format!("a{k},w{who},c_{},{t}\n", 1 + r.below(5))),
                        3 => clicks.push_str(&format!("z{k},w{who},c_{},{t}\n", 1 + r.below(5))),
                        _ => {}
                    }
                }
            }
        }
        std::fs::create_dir_all(dir.join("events")).unwrap();
        std::fs::write(dir.join("events/part-0.csv"), events).unwrap();
        std::fs::create_dir_all(dir.join("clicks")).unwrap();
        std::fs::write(dir.join("clicks/part-0.csv"), clicks).unwrap();
        std::fs::create_dir_all(dir.join("links")).unwrap();
        std::fs::write(dir.join("links/part-0.csv"), "user_id_a,user_id_b\nw38,w39\n").unwrap();
        let stats = compare(&dir, &cfg(2, 4096, 1));
        credited += stats[3];
        assists += stats[4];
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(credited > 2_000 && assists > credited, "credited {credited}, assists {assists}");
    let _ = std::fs::remove_dir_all(&root);
}

/// A field, quoted one time in four.
fn quoted(r: &mut Rng, field: &str) -> String {
    if r.chance(4) {
        format!("\"{field}\"")
    } else {
        field.to_string()
    }
}

fn shuffle<T>(r: &mut Rng, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        items.swap(i, r.below(i as u64 + 1) as usize);
    }
}

/// Copies of one event in every arrangement dedup must get right, in the
/// one-pass shape (person records marked dropped in place) and the
/// three-phase one: copies naming two IDs of one person or two persons
/// (so two person partitions); an earliest copy that is invalid in each
/// way (empty user, bad ts, bad type, empty product, bad price, late) with
/// later valid copies; a later invalid copy; equal ingest_ts; ingests that
/// are not whole seconds, far from the window or not timestamps (the exact
/// out-of-line path); a key with thousands of copies; copies spread over
/// files and chunks and read in any order. Users sit at the end of a link
/// chain hundreds of links long, and clicks come from IDs with no events.
#[test]
fn duplicate_copies_match_reference() {
    let root = std::env::temp_dir().join(format!("sessions-dups-{}", std::process::id()));
    let mut sessions = 0;
    for seed in 0..8u64 {
        let mut r = Rng(seed + 3000);
        let dir = root.join(format!("d{seed}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ws = days_from_civil(2026, 10, 1) * DAY_MS;
        let we = ws + 14 * DAY_MS;
        std::fs::write(
            dir.join("manifest.json"),
            format!("{{\"window_start\": \"{}Z\", \"window_end\": \"{}Z\"}}\n", utc(ws), utc(we)),
        )
        .unwrap();
        std::fs::write(dir.join("products.csv"), "product_id,category\np1,shoes\np2,books\np3,toys\n").unwrap();
        std::fs::write(dir.join("campaigns.csv"), "campaign_id,channel\nc_1,search\nc_2,social\nc_3,email\n").unwrap();

        // People: some with two or three linked IDs, one chain of 400 IDs
        // linked one after the next (named by an upper-case ID that only
        // the links know), and people of one ID.
        let mut links: Vec<String> = Vec::new();
        let mut persons: Vec<Vec<String>> = Vec::new();
        for j in 0..30 {
            let ids: Vec<String> = (0..2 + r.below(2)).map(|i| format!("m{j}x{i}")).collect();
            for w in ids.windows(2) {
                links.push(format!("{},{}", w[1], w[0]));
            }
            persons.push(ids);
        }
        let chain: Vec<String> = (0..400).map(|k| format!("ch{k:03}")).collect();
        for k in 0..399 {
            links.push(format!("{},{}", chain[k + 1], chain[k]));
        }
        links.push(format!("{},Ach", chain[399]));
        persons.push(chain[300..].to_vec());
        for j in 0..40 {
            persons.push(vec![format!("s{j}")]);
        }
        // An ID linked to a person but used only by clicks.
        links.push("m0x9,m0x0".to_string());
        shuffle(&mut r, &mut links);

        let user = |r: &mut Rng| -> (usize, String) {
            let p = r.below(persons.len() as u64) as usize;
            (p, persons[p][r.below(persons[p].len() as u64) as usize].clone())
        };
        let mut lines: Vec<String> = Vec::new();
        let line = |r: &mut Rng, id: &str, user: &str, ts: &str, ingest: &str, kind: &str, prod: &str, price: &str| -> String {
            [id, user, ts, ingest, kind, prod, price].iter().map(|f| quoted(r, f)).collect::<Vec<_>>().join(",")
        };
        for n in 0..3000u64 {
            let id = format!("e{n}");
            let t = ws + r.below((we - ws) as u64 - 2 * DAY_MS as u64) as i64 + DAY_MS;
            let ts = if r.chance(2) { format!("{t}") } else { format!("{}Z", utc(t - t.rem_euclid(1000))) };
            let t = if ts.ends_with('Z') { t - t.rem_euclid(1000) } else { t };
            let kind = if r.chance(4) { "purchase" } else if r.chance(2) { "cart" } else { "view" };
            let prod = format!("p{}", 1 + r.below(4));
            let price = format!("{}", 100 + r.below(900));
            let (p, u) = user(&mut r);
            let alt = user(&mut r).1;
            // Whole seconds after ts, up to 10 minutes.
            let ingest = t - t.rem_euclid(1000) + 1000 * (1 + r.below(600) as i64);
            let f1 = |ms: i64| format!("{}Z", utc(ms));
            let mut copies: Vec<String> = Vec::new();
            match n % 10 {
                0 => copies.push(line(&mut r, &id, &u, &ts, &f1(ingest), kind, &prod, &price)),
                1 => {
                    // Identical copies, later ingests.
                    for c in 0..2 + r.below(3) as i64 {
                        let at = ingest + c * 1000 * (1 + r.below(600) as i64);
                        copies.push(line(&mut r, &id, &u, &ts, &f1(at), kind, &prod, &price));
                    }
                }
                2 => {
                    // Later copies name another ID of the same person, or
                    // another person; any of them may be the earliest.
                    let other = if r.chance(2) && persons[p].len() > 1 {
                        persons[p][r.below(persons[p].len() as u64) as usize].clone()
                    } else {
                        user(&mut r).1
                    };
                    let mut at = ingest;
                    for c in 0..2 + r.below(3) {
                        let who = if (c + n) % 2 == 0 { &u } else { &other };
                        copies.push(line(&mut r, &id, who, &ts, &f1(at), kind, &prod, &price));
                        at += 1000 * (1 + r.below(300) as i64);
                    }
                }
                3 => {
                    // The earliest copy is invalid in one way; later copies
                    // are valid. The event must be dropped.
                    let first = match n / 10 % 6 {
                        0 => line(&mut r, &id, "", &ts, &f1(ingest), kind, &prod, &price),
                        1 => line(&mut r, &id, &u, "2026-13-03T10:00:00Z", &f1(ingest), kind, &prod, &price),
                        2 => line(&mut r, &id, &u, &ts, &f1(ingest), "Purchase", &prod, &price),
                        3 => line(&mut r, &id, &u, &ts, &f1(ingest), kind, "", &price),
                        4 => line(&mut r, &id, &u, &ts, &f1(ingest), "purchase", &prod, "12.5"),
                        // Late: its own ts is two days before its ingest.
                        _ => line(&mut r, &id, &u, &format!("{}", ingest - 2 * DAY_MS), &f1(ingest), kind, &prod, &price),
                    };
                    copies.push(first);
                    for c in 1..2 + r.below(3) as i64 {
                        copies.push(line(&mut r, &id, &u, &ts, &f1(ingest + c * 1000), kind, &prod, &price));
                    }
                }
                4 => {
                    // Equal ingests: identical copies, quoted differently,
                    // and a later copy.
                    for _ in 0..2 + r.below(3) {
                        copies.push(line(&mut r, &id, &u, &ts, &f1(ingest), kind, &prod, &price));
                    }
                    copies.push(line(&mut r, &id, &alt, &ts, &f1(ingest + 1000), kind, &prod, &price));
                }
                5 => {
                    // Ingests that are not whole seconds (F3 with
                    // milliseconds), F2, and F1, a millisecond apart or
                    // equal; equal ones are identical.
                    let mut at = ingest - 1000;
                    for c in 0..2 + r.below(4) {
                        let field = match (c + n / 10) % 3 {
                            0 => format!("{at}"),
                            1 => format!("{}+01:00", utc(at - at.rem_euclid(1000) + 3_600_000)),
                            _ => f1(at),
                        };
                        // One user: two forms may name the same instant, and copies
                        // with equal ingests must be identical.
                        let who = u.clone();
                        copies.push(line(&mut r, &id, &who, &ts, &field, kind, &prod, &price));
                        at += [1, 999, 1000, 1001][r.below(4) as usize];
                    }
                }
                6 => {
                    // An ingest that is no timestamp sorts last: with a
                    // valid copy, or alone.
                    copies.push(line(&mut r, &id, &u, &ts, "garbage", kind, &prod, &price));
                    if r.chance(2) {
                        copies.push(line(&mut r, &id, &u, &ts, &f1(ingest), kind, &prod, &price));
                    }
                }
                7 => {
                    // A valid earliest copy and later invalid ones: kept.
                    copies.push(line(&mut r, &id, &u, &ts, &f1(ingest), kind, &prod, &price));
                    copies.push(line(&mut r, &id, "", &ts, &f1(ingest + 1000), kind, &prod, &price));
                    copies.push(line(&mut r, &id, &u, &ts, &f1(ingest + 2000), "Cart", &prod, &price));
                }
                8 => {
                    // Ingests far from the window (outside the note clock):
                    // decades before, which wins, and decades after.
                    let far = if r.chance(2) { days_from_civil(1940, 5, 1) * DAY_MS } else { days_from_civil(2200, 1, 1) * DAY_MS };
                    copies.push(line(&mut r, &id, &u, &ts, &f1(far + 7000), kind, &prod, &price));
                    copies.push(line(&mut r, &id, &alt, &ts, &f1(ingest), kind, &prod, &price));
                    copies.push(line(&mut r, &id, &u, &ts, &format!("{}", far + 7001), kind, &prod, &price));
                }
                _ => {
                    // Copies with milliseconds against whole seconds.
                    copies.push(line(&mut r, &id, &u, &ts, &format!("{}", ingest + 1), kind, &prod, &price));
                    copies.push(line(&mut r, &id, &alt, &ts, &f1(ingest), kind, &prod, &price));
                }
            }
            lines.extend(copies);
        }
        // One key with thousands of copies, all ingests distinct, users all
        // over; the earliest copy is valid on even seeds and invalid on odd.
        let hot_t = ws + 3 * DAY_MS + 12_345;
        let mut order: Vec<i64> = (0..3000).collect();
        shuffle(&mut r, &mut order);
        for c in order {
            let kind = if c == 0 && seed % 2 == 1 { "Purchase" } else if c % 7 == 3 { "bogus" } else { "view" };
            let u = user(&mut r).1;
            lines.push(line(&mut r, "hot", &u, &format!("{hot_t}"), &format!("{}Z", utc(hot_t - 12_345 + 60_000 + c * 1000)), kind, "p1", ""));
        }
        shuffle(&mut r, &mut lines);
        let mut bodies: Vec<Vec<u8>> = vec![Vec::new(); 3];
        for l in &lines {
            let f = r.below(3) as usize;
            bodies[f].extend_from_slice(l.as_bytes());
            bodies[f].push(b'\n');
        }
        write_parts(&dir.join("events"), "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents", bodies, false);

        let mut clicks = String::new();
        for k in 0..600 {
            let who = match r.below(4) {
                0 => format!("g{}", r.below(20)),
                1 => "m0x9".to_string(),
                _ => user(&mut r).1,
            };
            let t = ws - 3 * DAY_MS + r.below(16 * DAY_MS as u64) as i64;
            clicks.push_str(&format!("k{k},{who},c_{},{t}\n", 1 + r.below(3)));
        }
        write_parts(&dir.join("clicks"), "click_id,user_id,campaign_id,ts", vec![clicks.into_bytes()], false);
        let half = links.len() / 2;
        let link_bodies = vec![links[..half].join("\n").into_bytes(), (links[half..].join("\n") + "\n").into_bytes()];
        write_parts(&dir.join("links"), "user_id_a,user_id_b", link_bodies.into_iter().map(|mut b| {
            if b.last() != Some(&b'\n') {
                b.push(b'\n');
            }
            b
        }).collect(), false);

        let (threads, chunk, bits) = match seed % 4 {
            0 => (1, 4 << 20, 1),
            1 => (4, 256, 4),
            2 => (3, 5000, 2),
            _ => (8, 1024, 3),
        };
        sessions += compare(&dir, &cfg(threads, chunk, bits))[2];
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(sessions > 10_000, "only {sessions} sessions");
    let _ = std::fs::remove_dir_all(&root);
}

fn dense_cfg(threads: usize, chunk: usize, bits: u32, slots: Option<usize>) -> Config {
    Config { dense: true, dense_slots: slots, ..cfg(threads, chunk, bits) }
}

fn write_dataset(dir: &Path, ws: i64, days: i64, events: &str, clicks: &str, links: &str) {
    let _ = std::fs::remove_dir_all(dir);
    for sub in ["events", "clicks", "links"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    std::fs::write(
        dir.join("manifest.json"),
        format!("{{\"window_start\": \"{}Z\", \"window_end\": \"{}Z\"}}\n", utc(ws), utc(ws + days * DAY_MS)),
    )
    .unwrap();
    std::fs::write(dir.join("products.csv"), "product_id,category\np1,shoes\np2,books\np3,toys\n").unwrap();
    std::fs::write(dir.join("campaigns.csv"), "campaign_id,channel\nc_1,search\nc_2,social\nc_3,email\n").unwrap();
    let header = "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents";
    std::fs::write(dir.join("events/part-0.csv"), format!("{header}\n{events}")).unwrap();
    std::fs::write(dir.join("clicks/part-0.csv"), format!("click_id,user_id,campaign_id,ts\n{clicks}")).unwrap();
    std::fs::write(dir.join("links/part-0.csv"), format!("user_id_a,user_id_b\n{links}")).unwrap();
}

/// The dense shape with ID tables from far too small (it must give up and
/// the three-phase shape must answer) to roomy, and person partitions of
/// four slots up, so the index width is small and many partitions share
/// the persons. Exact in every case.
#[test]
fn dense_table_sizes_match_reference() {
    let root = std::env::temp_dir().join(format!("sessions-dense-sizes-{}", std::process::id()));
    let (mut ran, mut gave_up, mut tiny) = (0, 0, 0);
    for seed in 0..12u64 {
        let dir = root.join(format!("s{seed}"));
        generate(&dir, seed + 500, [300, 3000][(seed % 2) as usize]);
        if seed % 3 == 0 {
            add_long_lines(&dir, seed);
        }
        let lk = load_lookups(&dir);
        let want = naive::run(&dir, &lk);
        for slots in [64usize, 256, 1024, 4096, 16384, 1 << 16] {
            for (threads, chunk, bits) in [(4, 256, 6), (3, 5000, 2), (8, 1024, 10)] {
                let config = dense_cfg(threads, chunk, bits, Some(slots));
                if check_shape(&dir, &config, &lk, &want) {
                    ran += 1;
                    tiny += usize::from(slots >> bits.min(slots.trailing_zeros() - 2) <= 8);
                } else {
                    gave_up += 1;
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(ran > 50 && gave_up > 20 && tiny > 10, "ran {ran}, gave up {gave_up}, tiny partitions {tiny}");
    let _ = std::fs::remove_dir_all(&root);
}

/// A click record holds `index << cbits | campaign` in 32 bits. When the
/// index and the campaign do not fit, the dense shape gives up before it
/// touches anything, and the answer is still exact.
#[test]
fn dense_wide_index_gives_up() {
    let dir = std::env::temp_dir().join(format!("sessions-dense-wide-{}", std::process::id()));
    generate(&dir, 77, 300);
    let lk = load_lookups(&dir);
    let want = naive::run(&dir, &lk);
    assert!(!check_shape(&dir, &dense_cfg(2, 4096, 1, Some(1 << 31)), &lk, &want));
    assert!(check_shape(&dir, &dense_cfg(2, 4096, 1, Some(1 << 12)), &lk, &want));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Few IDs, many rows, many threads and tiny chunks: threads keep meeting
/// the same new IDs at the same moment. IDs come first from events, from
/// clicks, or only from links; a person may be named by an ID no row uses,
/// and IDs that cannot be packed are linked to ones that can.
#[test]
fn dense_new_ids_race_and_match() {
    let root = std::env::temp_dir().join(format!("sessions-dense-race-{}", std::process::id()));
    for seed in 0..6u64 {
        let mut r = Rng(seed + 9000);
        let dir = root.join(format!("r{seed}"));
        let ws = days_from_civil(2026, 10, 1) * DAY_MS;
        let users: Vec<String> = (0..24)
            .map(|i| match i % 6 {
                0 => format!("averyveryverylonguser{i}"),
                1 => format!("u\u{00e9}{i}"),
                _ => format!("u{i}"),
            })
            .collect();
        let mut events = String::new();
        for n in 0..6000 {
            let t = ws + r.below(13 * DAY_MS as u64) as i64;
            let ingest = t - t.rem_euclid(1000) + 1000 * (1 + r.below(100) as i64);
            let kind = if r.chance(4) { "purchase" } else { "view" };
            let u = &users[r.below(users.len() as u64) as usize];
            events.push_str(&format!("e{n},{u},{t},{}Z,{kind},p{},{}\n", utc(ingest), 1 + r.below(3), 100 + r.below(900)));
            if r.chance(20) {
                // A later copy naming another user.
                let v = &users[r.below(users.len() as u64) as usize];
                events.push_str(&format!("e{n},{v},{t},{}Z,{kind},p1,500\n", utc(ingest + 5000)));
            }
        }
        let mut clicks = String::new();
        for k in 0..2000 {
            // Click-only IDs too: never in an event.
            let u = if r.chance(5) { format!("konly{}", r.below(5)) } else { users[r.below(users.len() as u64) as usize].clone() };
            let t = ws - 2 * DAY_MS + r.below(15 * DAY_MS as u64) as i64;
            clicks.push_str(&format!("k{k},{u},c_{},{t}\n", 1 + r.below(3)));
        }
        // u2..u5 are named by "Aname", which no row uses; the long ID and
        // the non-ASCII ones are linked to packable IDs; click-only IDs are
        // linked to event IDs; some IDs are only in links.
        let mut links = String::from("u2,Aname\nu3,u2\nu4,u3\nu5,Aname\n");
        links.push_str("averyveryverylonguser0,u7\nu\u{00e9}1,u8\naveryveryverylonguser6,u\u{00e9}7\n");
        links.push_str("konly0,u9\nkonly1,u10\nlinkonly1,u11\nlinkonly2,linkonly3\nu12,zzlink\n");
        write_dataset(&dir, ws, 14, &events, &clicks, &links);
        let lk = load_lookups(&dir);
        let want = naive::run(&dir, &lk);
        for (threads, chunk) in [(16, 64), (12, 128), (32, 200), (7, 333)] {
            assert!(check_shape(&dir, &dense_cfg(threads, chunk, 3, None), &lk, &want), "seed {seed}: dense gave up");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// One person with hundreds of thousands of events over several linked
/// IDs, with clicks and purchases, next to small people.
#[test]
fn dense_whale_matches_reference() {
    let dir = std::env::temp_dir().join(format!("sessions-dense-whale-{}", std::process::id()));
    let mut r = Rng(4242);
    let ws = days_from_civil(2027, 1, 1) * DAY_MS;
    let mut events = String::new();
    for n in 0..300_000u64 {
        let who = if n % 50 == 0 { format!("s{}", r.below(300)) } else { format!("w{}", r.below(5)) };
        let t = ws + r.below(14 * DAY_MS as u64) as i64;
        let kind = if r.chance(30) { "purchase" } else { "view" };
        let ingest = t - t.rem_euclid(1000) + 2000;
        events.push_str(&format!("e{n},{who},{t},{}Z,{kind},p{},{}\n", utc(ingest), 1 + r.below(3), 1 + r.below(5000)));
        if r.chance(40) {
            events.push_str(&format!("e{n},s1,{t},{}Z,view,p1,\n", utc(ingest + 1000)));
        }
    }
    let mut clicks = String::new();
    for k in 0..5000 {
        let who = if k % 3 == 0 { format!("s{}", r.below(300)) } else { format!("w{}", r.below(5)) };
        let t = ws - 7 * DAY_MS + r.below(21 * DAY_MS as u64) as i64;
        clicks.push_str(&format!("k{k},{who},c_{},{t}\n", 1 + r.below(3)));
    }
    write_dataset(&dir, ws, 14, &events, &clicks, "w1,w0\nw2,w1\nw3,w4\nw4,w0\n");
    let lk = load_lookups(&dir);
    let want = naive::run(&dir, &lk);
    let biggest: i64 = want.user.iter().filter(|r| r.user_id == b"w0").map(|r| r.events).sum();
    assert!(biggest > 250_000, "whale has {biggest} events");
    for (threads, chunk, bits) in [(8, 1 << 16, 3), (4, 4 << 20, 1)] {
        assert!(check_shape(&dir, &dense_cfg(threads, chunk, bits, None), &lk, &want));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Times are 32-bit milliseconds from the click window start. A window of
/// 42 days (49 with the click reach) fits and runs dense; 43 days does not,
/// and the fast engine declines, so the reference answers.
#[test]
fn dense_long_window_falls_back() {
    let root = std::env::temp_dir().join(format!("sessions-dense-long-{}", std::process::id()));
    let ws = days_from_civil(2026, 1, 1) * DAY_MS;
    for days in [42i64, 43] {
        let dir = root.join(format!("d{days}"));
        let mut r = Rng(days as u64);
        let mut events = String::new();
        for n in 0..3000 {
            let t = ws + r.below((days * DAY_MS) as u64) as i64;
            events.push_str(&format!("e{n},u{},{t},{}Z,view,p1,\n", r.below(40), utc(t - t.rem_euclid(1000) + 1000)));
        }
        let mut clicks = String::new();
        for k in 0..500 {
            let t = ws - 7 * DAY_MS + r.below(((days + 7) * DAY_MS) as u64) as i64;
            clicks.push_str(&format!("k{k},u{},c_1,{t}\n", r.below(40)));
        }
        write_dataset(&dir, ws, days, &events, &clicks, "u1,u2\n");
        let lk = load_lookups(&dir);
        let config = dense_cfg(4, 4096, 2, None);
        if days == 42 {
            let want = naive::run(&dir, &lk);
            assert!(check_shape(&dir, &config, &lk, &want));
        } else {
            assert!(engine::run(&dir, None, &lk, &config, std::time::Instant::now()).is_none());
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// The click parser writes into a block of `CLICK_BLOCK` clicks through a
/// raw cursor; the long-line path pushes into the same block between
/// windows. Bound: a block never holds more than `CLICK_BLOCK` clicks and
/// its length is exact after every window. Here: one thread, one chunk,
/// more than two blocks of clicks, long click lines around each block end,
/// and the result against the reference.
#[test]
fn click_blocks_fill_across_long_lines() {
    use crate::engine::CLICK_BLOCK;
    let root = std::env::temp_dir().join(format!("sessions-clickblock-{}", std::process::id()));
    let dir = root.join("d");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ws = days_from_civil(2027, 3, 1) * DAY_MS;
    let we = ws + 14 * DAY_MS;
    std::fs::write(dir.join("manifest.json"), format!("{{\"window_start\": \"{}Z\", \"window_end\": \"{}Z\"}}\n", utc(ws), utc(we))).unwrap();
    std::fs::write(dir.join("products.csv"), "product_id,category\np1,shoes\n").unwrap();
    std::fs::write(dir.join("campaigns.csv"), "campaign_id,channel\nc_1,search\nc_2,social\nc_3,email\n").unwrap();
    let users = 300u64;
    let mut events = String::from("event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
    for u in 0..users {
        for j in 0..3i64 {
            let t = ws + (u as i64 * 7_919_000 + j * 40_000_000) % (we - ws - DAY_MS);
            events.push_str(&format!("e{u}x{j},w{u},{t},{}Z,view,p1,\n", utc(t - t.rem_euclid(1000) + 1000)));
        }
    }
    let total = 2 * CLICK_BLOCK + 3000;
    let mut clicks = String::from("click_id,user_id,campaign_id,ts\n");
    let mut long = 0;
    for k in 0..total {
        // Long lines just before, at and after each block end.
        if (k + 2) % CLICK_BLOCK < 5 {
            let id: String = std::iter::repeat_n('k', 70_000 + k % 7).collect();
            clicks.push_str(&format!("{id},w{},c_{},{}\n", k as u64 % users, 1 + k % 3, ws + (k as i64 * 37_000) % (we - ws)));
            long += 1;
        }
        let t = ws - 3 * DAY_MS + (k as i64 * 61_000) % (we - ws);
        clicks.push_str(&format!("q{k},w{},c_{},{t}\n", (k as u64 * 7) % users, 1 + k % 3));
    }
    assert!(long >= 10);
    std::fs::create_dir_all(dir.join("events")).unwrap();
    std::fs::write(dir.join("events/part-0.csv"), events).unwrap();
    std::fs::create_dir_all(dir.join("clicks")).unwrap();
    std::fs::write(dir.join("clicks/part-0.csv"), clicks).unwrap();
    std::fs::create_dir_all(dir.join("links")).unwrap();
    std::fs::write(dir.join("links/part-0.csv"), "user_id_a,user_id_b\nw1,w2\n").unwrap();
    let a = compare(&dir, &cfg(1, 64 << 20, 1));
    assert_eq!(compare(&dir, &cfg(2, 300_000, 2)), a);
    assert!(a[3] > 100, "credited {}", a[3]);
    let _ = std::fs::remove_dir_all(&root);
}
