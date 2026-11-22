//! Shared pieces: the lookup files, the output row types, and the plain
//! (slow, obviously correct) CSV and timestamp readers that the naive
//! reference uses and that every fast path is tested against.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const DAY_MS: i64 = 86_400_000;
pub const SESSION_GAP_MS: i64 = 1_800_000;
pub const CLICK_REACH_MS: i64 = 7 * DAY_MS;
pub const LATE_MS: i64 = DAY_MS;
pub const MAX_PRICE: u64 = 10_000_000;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UserRow {
    pub user_id: Vec<u8>,
    pub day: i32,
    pub sessions: i64,
    pub events: i64,
    pub revenue: i64,
    pub top: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CampRow {
    pub campaign: Vec<u8>,
    pub channel: Vec<u8>,
    pub day: i32,
    pub sessions: i64,
    pub purchasing: i64,
    pub revenue: i64,
    pub assisted: i64,
}

#[derive(Debug, Default)]
pub struct Tables {
    pub user: Vec<UserRow>,
    pub camp: Vec<CampRow>,
    /// The fast engine ran its dense shape to the end (tests).
    pub dense: bool,
}

impl Tables {
    pub fn sort(&mut self) {
        self.user.sort();
        self.camp.sort();
    }
}

/// Everything read before the big files: the window and the two catalogs.
pub struct Lookups {
    pub window_start: i64,
    pub window_end: i64,
    /// Category names, sorted byte-wise. Index is the category id, so the
    /// smaller id wins a tie. `unknown` is always present.
    pub cats: Vec<Vec<u8>>,
    pub unknown_cat: u32,
    /// Category id of product `p<n>` (canonical decimal `n`), by `n`;
    /// `u32::MAX` where the file has no such product.
    pub product_dense: Vec<u32>,
    /// Every other product id in the file, exact bytes, to its category id.
    pub product_odd: HashMap<Vec<u8>, u32>,
    /// Campaign ids and channels in file order. Index is the campaign id.
    pub camp_ids: Vec<Vec<u8>>,
    pub camp_channels: Vec<Vec<u8>>,
    pub campaigns: HashMap<Vec<u8>, u32, IdHash>,
}

impl Lookups {
    /// The category of a product id, `unknown` when the file has none.
    pub fn product_cat(&self, id: &[u8]) -> u32 {
        let found = match canonical_suffix(id, b"p") {
            Some(n) => self.product_dense.get(n as usize).copied().filter(|&c| c != u32::MAX),
            None => self.product_odd.get(id).copied(),
        };
        found.unwrap_or(self.unknown_cat)
    }
}

/// `<prefix><n>` with `n` a decimal of 1 to 7 digits and no leading zero:
/// the one spelling of that id, so it can index a dense table.
pub fn canonical_suffix(id: &[u8], prefix: &[u8]) -> Option<u32> {
    let rest = id.strip_prefix(prefix)?;
    if rest.is_empty() || rest.len() > 7 || rest[0] == b'0' || !rest.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(rest.iter().fold(0u32, |acc, &c| acc * 10 + u32::from(c - b'0')))
}

pub fn die(msg: &str) -> ! {
    eprintln!("sessions: {msg}");
    std::process::exit(1);
}

pub fn list_csv(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".csv") && path.is_file() {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// Split one line into `n` fields under the rules' dialect. `None` when the
/// line is malformed: wrong field count, or a quote anywhere except as a
/// pair around a whole field.
pub fn split_line<'a>(line: &'a [u8], n: usize, out: &mut Vec<&'a [u8]>) -> bool {
    out.clear();
    for field in line.split(|&b| b == b',') {
        let quotes = field.iter().filter(|&&b| b == b'"').count();
        let value = if quotes == 0 {
            field
        } else if quotes == 2 && field.len() >= 2 && field[0] == b'"' && field[field.len() - 1] == b'"'
        {
            &field[1..field.len() - 1]
        } else {
            return false;
        };
        out.push(value);
    }
    out.len() == n
}

/// Data lines of a file: everything after the header line, split on `\n`
/// only. A final line without a newline still counts; the empty remainder
/// after a final newline does not.
pub fn data_lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    let body = match bytes.iter().position(|&b| b == b'\n') {
        Some(at) => &bytes[at + 1..],
        None => &bytes[bytes.len()..],
    };
    let mut rest = body;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        match rest.iter().position(|&b| b == b'\n') {
            Some(at) => {
                let line = &rest[..at];
                rest = &rest[at + 1..];
                Some(line)
            }
            None => {
                let line = rest;
                rest = &rest[rest.len()..];
                Some(line)
            }
        }
    })
}

pub fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

pub fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if is_leap(year) {
                29
            } else {
                28
            }
        }
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
pub fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn two(b: &[u8]) -> Option<i64> {
    if b[0].is_ascii_digit() && b[1].is_ascii_digit() {
        Some(i64::from(b[0] - b'0') * 10 + i64::from(b[1] - b'0'))
    } else {
        None
    }
}

/// `YYYY-MM-DDTHH:MM:SS` (19 bytes) as local milliseconds.
fn civil_ms(b: &[u8]) -> Option<i64> {
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let year = two(&b[0..2])? * 100 + two(&b[2..4])?;
    let month = two(&b[5..7])?;
    let day = two(&b[8..10])?;
    let hour = two(&b[11..13])?;
    let minute = two(&b[14..16])?;
    let second = two(&b[17..19])?;
    if !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    Some((days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second) * 1_000)
}

/// The reference timestamp reader: the three forms of the rules, nothing
/// else. A digits-only value too large to matter returns `i64::MAX`, which
/// is outside every window.
pub fn parse_ts_naive(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    if b.iter().all(|c| c.is_ascii_digit()) {
        let digits: &[u8] = {
            let lead = b.iter().take_while(|&&c| c == b'0').count();
            &b[lead..]
        };
        if digits.len() > 18 {
            return Some(i64::MAX);
        }
        let mut value = 0i64;
        for &c in digits {
            value = value * 10 + i64::from(c - b'0');
        }
        return Some(value);
    }
    if b.len() == 20 && b[19] == b'Z' {
        return civil_ms(&b[..19]);
    }
    if b.len() == 25 && (b[19] == b'+' || b[19] == b'-') && b[22] == b':' {
        let local = civil_ms(&b[..19])?;
        let hours = two(&b[20..22])?;
        let minutes = two(&b[23..25])?;
        if hours > 23 || minutes > 59 {
            return None;
        }
        let offset = (hours * 60 + minutes) * 60_000;
        return Some(if b[19] == b'+' { local - offset } else { local + offset });
    }
    None
}

/// A purchase price: ASCII digits only, 1 to 10,000,000.
pub fn parse_price_naive(b: &[u8]) -> Option<u64> {
    if b.is_empty() || !b.iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let lead = b.iter().take_while(|&&c| c == b'0').count();
    let digits = &b[lead..];
    if digits.len() > 8 {
        return None;
    }
    let mut value = 0u64;
    for &c in digits {
        value = value * 10 + u64::from(c - b'0');
    }
    (1..=MAX_PRICE).contains(&value).then_some(value)
}

fn manifest_value(text: &[u8], key: &str) -> Option<i64> {
    let needle = format!("\"{key}\"");
    let at = text
        .windows(needle.len())
        .position(|w| w == needle.as_bytes())?;
    let rest = &text[at + needle.len()..];
    let open = rest.iter().position(|&b| b == b'"')?;
    let rest = &rest[open + 1..];
    let close = rest.iter().position(|&b| b == b'"')?;
    parse_ts_naive(&rest[..close])
}

/// The two fields of every data line of a two-column catalog, as
/// `data_lines` and `split_line(line, 2, _)` give them, skipping malformed
/// lines. A file without quotes (the usual case) is split in one pass over
/// its bytes; any quote sends the whole file through the reference readers.
fn for_each_pair<'a>(bytes: &'a [u8], mut f: impl FnMut(&'a [u8], &'a [u8])) {
    if bytes.contains(&b'"') {
        let mut fields = Vec::new();
        for line in data_lines(bytes) {
            if split_line(line, 2, &mut fields) {
                f(fields[0], fields[1]);
            }
        }
        return;
    }
    let Some(header) = bytes.iter().position(|&b| b == b'\n') else {
        return;
    };
    let mut start = header + 1;
    let (mut comma, mut commas) = (0usize, 0u32);
    // Only commas and line breaks matter: visit them a 64-byte block at a
    // time, from bit masks, and skip the bytes between.
    let mut at = start;
    while at < bytes.len() {
        let n = (bytes.len() - at).min(64);
        let (mut commas_m, newlines_m) = delimiter_masks(&bytes[at..at + n]);
        let mut m = commas_m | newlines_m;
        while m != 0 {
            let bit = m.trailing_zeros();
            let i = at + bit as usize;
            if commas_m & 1 << bit != 0 {
                comma = i;
                commas += 1;
            } else {
                if commas == 1 {
                    f(&bytes[start..comma], &bytes[comma + 1..i]);
                }
                start = i + 1;
                commas = 0;
            }
            commas_m &= !(1 << bit);
            m &= m - 1;
        }
        at += n;
    }
    // A last line without a line break; the empty remainder after a final
    // line break is no line.
    if start < bytes.len() && commas == 1 {
        f(&bytes[start..comma], &bytes[comma + 1..]);
    }
}

/// Bit `i` set where `block[i]` is a comma, and where it is a line break.
/// `block` is at most 64 bytes.
#[inline(always)]
fn delimiter_masks(block: &[u8]) -> (u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if block.len() == 64 {
        unsafe {
            use core::arch::x86_64::*;
            let lo = _mm256_loadu_si256(block.as_ptr() as *const __m256i);
            let hi = _mm256_loadu_si256(block.as_ptr().add(32) as *const __m256i);
            let mask = |v: __m256i, b: u8| _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b as i8))) as u32 as u64;
            return (mask(lo, b',') | mask(hi, b',') << 32, mask(lo, b'\n') | mask(hi, b'\n') << 32);
        }
    }
    let (mut c, mut n) = (0u64, 0u64);
    for (i, &b) in block.iter().enumerate() {
        c |= u64::from(b == b',') << i;
        n |= u64::from(b == b'\n') << i;
    }
    (c, n)
}

/// A cheap hash for short ids (the campaign map is built on the clock):
/// eight bytes at a time, one multiply each. Equality is exact as always.
#[derive(Clone, Copy, Default)]
pub struct IdHash;

pub struct IdHasher(u64);

impl std::hash::BuildHasher for IdHash {
    type Hasher = IdHasher;
    fn build_hasher(&self) -> IdHasher {
        IdHasher(0)
    }
}

impl std::hash::Hasher for IdHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
        }
        let mut last = 0u64;
        for (i, &b) in chunks.remainder().iter().enumerate() {
            last |= u64::from(b) << (8 * i);
        }
        self.0 = (h ^ last ^ (bytes.len() as u64) << 56).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
    }
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 31)
    }
}

/// A catalog file, mapped and never unmapped (empty if it is missing).
/// Mapping it costs no copy and no fresh zeroed pages.
fn catalog_bytes(path: &Path) -> &'static [u8] {
    if !path.is_file() {
        return &[];
    }
    let m = crate::sys::map_file(path);
    unsafe { std::slice::from_raw_parts(m.ptr, m.len) }
}

/// Category names in order of first appearance, with an open-addressing
/// index: a byte-wise hash, exact compares.
struct NameTable<'a> {
    names: Vec<&'a [u8]>,
    /// Index of a name plus one; 0 is an empty slot.
    slots: Vec<u32>,
}

impl<'a> NameTable<'a> {
    /// Starts with `unknown` as name 0.
    fn new() -> Self {
        let mut t = NameTable { names: Vec::new(), slots: vec![0; 256] };
        t.index(b"unknown");
        t
    }

    /// Length and three bytes: constant time, and enough to spread a few
    /// hundred category names; equality is always decided by the bytes.
    fn hash(name: &[u8]) -> u64 {
        let n = name.len();
        let h = match n {
            0 => 0,
            _ => (n as u64) | u64::from(name[0]) << 16 | u64::from(name[n / 2]) << 24 | u64::from(name[n - 1]) << 32,
        };
        h.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    fn slot(&self, name: &[u8]) -> usize {
        let mask = self.slots.len() - 1;
        let mut i = (Self::hash(name) >> 32) as usize & mask;
        loop {
            let at = self.slots[i];
            if at == 0 || self.names[at as usize - 1] == name {
                return i;
            }
            i = (i + 1) & mask;
        }
    }

    /// The index of `name`, added if new.
    fn index(&mut self, name: &'a [u8]) -> usize {
        let i = self.slot(name);
        if self.slots[i] != 0 {
            return self.slots[i] as usize - 1;
        }
        self.names.push(name);
        let id = self.names.len() - 1;
        if self.names.len() * 2 > self.slots.len() {
            self.slots = vec![0; self.slots.len() * 2];
            for k in 0..self.names.len() {
                let at = self.slot(self.names[k]);
                self.slots[at] = k as u32 + 1;
            }
        } else {
            self.slots[i] = id as u32 + 1;
        }
        id
    }
}

pub fn load_lookups(data: &Path) -> Lookups {
    let manifest = std::fs::read(data.join("manifest.json"))
        .unwrap_or_else(|e| die(&format!("manifest.json: {e}")));
    let window_start = manifest_value(&manifest, "window_start")
        .unwrap_or_else(|| die("manifest.json: no window_start"));
    let window_end =
        manifest_value(&manifest, "window_end").unwrap_or_else(|| die("manifest.json: no window_end"));

    // Products. The catalog is large (50,000 rows) and loading it is on the
    // clock, so the common shape `p<n>` goes to a dense table and only odd
    // ids pay for a hash map entry. Both hold exact ids.
    let product_bytes = catalog_bytes(&data.join("products.csv"));
    let mut names = NameTable::new();
    let mut dense_raw: Vec<u32> = Vec::new();
    let mut odd_raw: Vec<(&[u8], u32)> = Vec::new();
    for_each_pair(product_bytes, |id, cat| {
        if id.is_empty() {
            return;
        }
        let idx = names.index(cat);
        match canonical_suffix(id, b"p") {
            Some(n) => {
                let n = n as usize;
                if dense_raw.len() <= n {
                    dense_raw.resize(n + 1, u32::MAX);
                }
                if dense_raw[n] == u32::MAX {
                    dense_raw[n] = idx as u32;
                }
            }
            None => odd_raw.push((id, idx as u32)),
        }
    });
    let names = names.names;
    // Category ids are ranks in byte order, so the smaller id wins a tie.
    let mut order: Vec<u32> = (0..names.len() as u32).collect();
    order.sort_by_key(|&i| names[i as usize]);
    let mut rank = vec![0u32; names.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i as usize] = r as u32;
    }
    let cats: Vec<Vec<u8>> = order.iter().map(|&i| names[i as usize].to_vec()).collect();
    let unknown_cat = rank[0];
    let product_dense: Vec<u32> = dense_raw.iter().map(|&i| if i == u32::MAX { u32::MAX } else { rank[i as usize] }).collect();
    let mut product_odd = HashMap::new();
    for (id, idx) in odd_raw {
        product_odd.entry(id.to_vec()).or_insert(rank[idx as usize]);
    }

    let campaign_bytes = catalog_bytes(&data.join("campaigns.csv"));
    let mut camp_ids = Vec::new();
    let mut camp_channels = Vec::new();
    let mut campaigns = HashMap::with_capacity_and_hasher(campaign_bytes.len() / 8, IdHash);
    for_each_pair(campaign_bytes, |id, channel| {
        if !id.is_empty() && !campaigns.contains_key(id) {
            campaigns.insert(id.to_vec(), camp_ids.len() as u32);
            camp_ids.push(id.to_vec());
            camp_channels.push(channel.to_vec());
        }
    });
    Lookups {
        window_start,
        window_end,
        cats,
        unknown_cat,
        product_dense,
        product_odd,
        camp_ids,
        camp_channels,
        campaigns,
    }
}

#[cfg(test)]
mod pair_tests {
    use super::*;

    #[test]
    fn for_each_pair_matches_split_line() {
        let alphabet = b"ab1,\n\"\r p_";
        let mut state = 0x1234_5678_9abc_def0u64;
        for round in 0..20_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let len = (state >> 55) as usize + round % 3;
            let mut bytes = Vec::new();
            for k in 0..len {
                let r = state.rotate_left(7 * k as u32 + 3) ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                // Mostly quote-free files, so the fast path runs.
                let mut b = alphabet[(r % alphabet.len() as u64) as usize];
                if b == b'"' && round % 4 != 0 {
                    b = b'a';
                }
                bytes.push(b);
            }
            let mut want = Vec::new();
            let mut fields = Vec::new();
            for line in data_lines(&bytes) {
                if split_line(line, 2, &mut fields) {
                    want.push((fields[0].to_vec(), fields[1].to_vec()));
                }
            }
            let mut got = Vec::new();
            for_each_pair(&bytes, |a, b| got.push((a.to_vec(), b.to_vec())));
            assert_eq!(got, want, "{:?}", String::from_utf8_lossy(&bytes));
        }
    }
}
