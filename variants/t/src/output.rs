//! The two output tables as Parquet, from plain rows. The fast engine
//! writes user_daily through its own column buffers (engine.rs) and uses
//! this module's schema; campaign_daily and the fallback path go through
//! the row writers here.

use std::path::Path;

use crate::common::*;
use crate::parquet::*;

pub const USER_SCHEMA: [ColSpec; 6] = [
    ColSpec { name: "user_id", phys: Phys::ByteArray, logical: Logical::Utf8, optional: false },
    ColSpec { name: "day", phys: Phys::Int32, logical: Logical::Date, optional: false },
    ColSpec { name: "sessions", phys: Phys::Int32, logical: Logical::None, optional: false },
    ColSpec { name: "events", phys: Phys::Int32, logical: Logical::None, optional: false },
    ColSpec { name: "revenue_cents", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "top_category", phys: Phys::ByteArray, logical: Logical::Utf8, optional: true },
];

pub const CAMP_SCHEMA: [ColSpec; 7] = [
    ColSpec { name: "campaign_id", phys: Phys::ByteArray, logical: Logical::Utf8, optional: false },
    ColSpec { name: "channel", phys: Phys::ByteArray, logical: Logical::Utf8, optional: false },
    ColSpec { name: "day", phys: Phys::Int32, logical: Logical::Date, optional: false },
    ColSpec { name: "sessions", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "purchasing_sessions", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "revenue_cents", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "assisted_sessions", phys: Phys::Int64, logical: Logical::None, optional: false },
];

/// The compact user_daily schema: 32-bit revenue. Rows that do not fit go
/// to a second file with `USER_SCHEMA`.
pub const USER_SCHEMA_COMPACT: [ColSpec; 6] = [
    USER_SCHEMA[0],
    USER_SCHEMA[1],
    USER_SCHEMA[2],
    USER_SCHEMA[3],
    ColSpec { name: "revenue_cents", phys: Phys::Int32, logical: Logical::None, optional: false },
    USER_SCHEMA[5],
];

/// Wide (64-bit counts) user_daily schema for the fallback path.
const USER_SCHEMA_WIDE: [ColSpec; 6] = [
    USER_SCHEMA[0],
    USER_SCHEMA[1],
    ColSpec { name: "sessions", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "events", phys: Phys::Int64, logical: Logical::None, optional: false },
    USER_SCHEMA[4],
    USER_SCHEMA[5],
];

pub fn make_dirs(out: &Path) {
    for sub in ["user_daily", "campaign_daily"] {
        if let Err(e) = std::fs::create_dir_all(out.join(sub)) {
            die(&format!("{}: {e}", out.join(sub).display()));
        }
    }
}

pub fn write_user_rows(path: &Path, rows: &[UserRow]) {
    let mut ids = Vec::new();
    let mut tops = Vec::new();
    for row in rows {
        plain_bytes(&row.user_id, &mut ids);
    }
    def_levels(rows.iter().map(|r| r.top.is_some()), rows.len(), &mut tops);
    for row in rows {
        if let Some(top) = &row.top {
            plain_bytes(top, &mut tops);
        }
    }
    let days: Vec<i32> = rows.iter().map(|r| r.day).collect();
    let sessions: Vec<i64> = rows.iter().map(|r| r.sessions).collect();
    let events: Vec<i64> = rows.iter().map(|r| r.events).collect();
    let revenue: Vec<i64> = rows.iter().map(|r| r.revenue).collect();
    let n = rows.len();
    let mut file = PqFile::create(path, &USER_SCHEMA_WIDE);
    file.write_row_group(
        n,
        &[
            vec![Page::plain(n, vec![&ids])],
            vec![Page::plain(n, vec![as_bytes(&days)])],
            vec![Page::plain(n, vec![as_bytes(&sessions)])],
            vec![Page::plain(n, vec![as_bytes(&events)])],
            vec![Page::plain(n, vec![as_bytes(&revenue)])],
            vec![Page::plain(n, vec![&tops])],
        ],
    );
    file.finish();
}

/// campaign_daily as columns. The text columns are runs: rows of one
/// campaign come together, so a run per campaign holds its id and channel.
#[derive(Default)]
pub struct CampCols<'a> {
    pub runs: Vec<(&'a [u8], &'a [u8], u32)>,
    pub day: Vec<i32>,
    pub sessions: Vec<i64>,
    pub purchasing: Vec<i64>,
    pub revenue: Vec<i64>,
    pub assisted: Vec<i64>,
}

impl<'a> CampCols<'a> {
    pub fn with_capacity(rows: usize) -> Self {
        CampCols {
            runs: Vec::new(),
            day: Vec::with_capacity(rows),
            sessions: Vec::with_capacity(rows),
            purchasing: Vec::with_capacity(rows),
            revenue: Vec::with_capacity(rows),
            assisted: Vec::with_capacity(rows),
        }
    }

    /// Start a run of rows of one campaign.
    pub fn campaign(&mut self, campaign: &'a [u8], channel: &'a [u8]) {
        self.runs.push((campaign, channel, 0));
    }

    /// A row of the campaign of the last run.
    pub fn push(&mut self, day: i32, sessions: i64, purchasing: i64, revenue: i64, assisted: i64) {
        self.runs.last_mut().expect("a campaign run").2 += 1;
        self.day.push(day);
        self.sessions.push(sessions);
        self.purchasing.push(purchasing);
        self.revenue.push(revenue);
        self.assisted.push(assisted);
    }

    pub fn len(&self) -> usize {
        self.day.len()
    }

    pub fn rows(&self) -> Vec<CampRow> {
        let mut out = Vec::with_capacity(self.len());
        let mut k = 0;
        for &(campaign, channel, count) in &self.runs {
            for _ in 0..count {
                out.push(CampRow {
                    campaign: campaign.to_vec(),
                    channel: channel.to_vec(),
                    day: self.day[k],
                    sessions: self.sessions[k],
                    purchasing: self.purchasing[k],
                    revenue: self.revenue[k],
                    assisted: self.assisted[k],
                });
                k += 1;
            }
        }
        out
    }

    pub fn of_rows(rows: &'a [CampRow]) -> Self {
        let mut cols = CampCols::with_capacity(rows.len());
        for (k, row) in rows.iter().enumerate() {
            if k == 0 || row.campaign != rows[k - 1].campaign || row.channel != rows[k - 1].channel {
                cols.campaign(&row.campaign, &row.channel);
            }
            cols.push(row.day, row.sessions, row.purchasing, row.revenue, row.assisted);
        }
        cols
    }
}

/// A text column given as runs, as a dictionary of its distinct values and
/// RLE runs of indices: the dictionary bytes, its size, the data page body.
fn dict_column<'a>(runs: impl Iterator<Item = (&'a [u8], u32)>) -> (Vec<u8>, usize, Vec<u8>) {
    let mut index: std::collections::HashMap<&[u8], u32> = std::collections::HashMap::new();
    let mut dict = Vec::new();
    let mut ids: Vec<(u32, u32)> = Vec::new();
    for (v, count) in runs {
        let next = index.len() as u32;
        let at = *index.entry(v).or_insert(next);
        if at == next {
            plain_bytes(v, &mut dict);
        }
        match ids.last_mut() {
            Some(last) if last.0 == at => last.1 += count,
            _ => ids.push((at, count)),
        }
    }
    let width = (usize::BITS - (index.len().max(2) - 1).leading_zeros()).max(1);
    let mut page = vec![width as u8];
    for &(at, count) in &ids {
        if count > 0 {
            rle_run(count as usize, at, width, &mut page);
        }
    }
    (dict, index.len(), page)
}

/// Write campaign_daily. campaign_id and channel are dictionary-encoded
/// (a few thousand runs instead of a string per row); the counts are plain.
pub fn write_camp_cols(path: &Path, cols: &CampCols) {
    let n = cols.len();
    let (id_dict, id_count, id_page) = dict_column(cols.runs.iter().map(|r| (r.0, r.2)));
    let (ch_dict, ch_count, ch_page) = dict_column(cols.runs.iter().map(|r| (r.1, r.2)));
    fn text<'b>(n: usize, dict: &'b [u8], count: usize, page: &'b [u8]) -> Vec<Page<'b>> {
        if n == 0 {
            vec![Page::plain(0, vec![])]
        } else {
            vec![Page::dict_page(count, vec![dict]), Page::dict_indices(n, vec![page])]
        }
    }
    let mut file = PqFile::create(path, &CAMP_SCHEMA);
    file.write_row_group(
        n,
        &[
            text(n, &id_dict, id_count, &id_page),
            text(n, &ch_dict, ch_count, &ch_page),
            vec![Page::plain(n, vec![as_bytes(&cols.day)])],
            vec![Page::plain(n, vec![as_bytes(&cols.sessions)])],
            vec![Page::plain(n, vec![as_bytes(&cols.purchasing)])],
            vec![Page::plain(n, vec![as_bytes(&cols.revenue)])],
            vec![Page::plain(n, vec![as_bytes(&cols.assisted)])],
        ],
    );
    file.finish();
}

pub fn write_tables(out: &Path, tables: &Tables) {
    make_dirs(out);
    write_user_rows(&out.join("user_daily/part-0.parquet"), &tables.user);
    write_camp_cols(&out.join("campaign_daily/part-0.parquet"), &CampCols::of_rows(&tables.camp));
}
