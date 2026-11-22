//! Just enough Parquet to write the marker rows: one row group, PLAIN
//! pages, no compression, the compact user_daily schema of the engine
//! (int32 counts and revenue, optional top_category). The Thrift encoder is
//! the engine's own (entry/src/parquet.rs).

const ENC_PLAIN: i32 = 0;
const ENC_RLE: i32 = 3;
const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_BINARY: u8 = 8;
const T_LIST: u8 = 9;
const T_STRUCT: u8 = 12;

struct Thrift {
    buf: Vec<u8>,
    last: Vec<i16>,
    cur: i16,
}

impl Thrift {
    fn new() -> Self {
        Thrift { buf: Vec::with_capacity(256), last: Vec::new(), cur: 0 }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }
    fn zigzag(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn field(&mut self, id: i16, ty: u8) {
        let delta = id - self.cur;
        if (1..=15).contains(&delta) {
            self.buf.push(((delta as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            self.zigzag(i64::from(id));
        }
        self.cur = id;
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, T_I32);
        self.zigzag(i64::from(v));
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, T_I64);
        self.zigzag(v);
    }
    fn string(&mut self, id: i16, s: &[u8]) {
        self.field(id, T_BINARY);
        self.varint(s.len() as u64);
        self.buf.extend_from_slice(s);
    }
    fn begin_struct(&mut self, id: i16) {
        self.field(id, T_STRUCT);
        self.last.push(self.cur);
        self.cur = 0;
    }
    fn end_struct(&mut self) {
        self.buf.push(0);
        self.cur = self.last.pop().unwrap_or(0);
    }
    fn begin_list(&mut self, id: i16, elem: u8, len: usize) {
        self.field(id, T_LIST);
        if len < 15 {
            self.buf.push(((len as u8) << 4) | elem);
        } else {
            self.buf.push(0xF0 | elem);
            self.varint(len as u64);
        }
    }
    fn begin_elem(&mut self) {
        self.last.push(self.cur);
        self.cur = 0;
    }
    fn elem_i32(&mut self, v: i32) {
        self.zigzag(i64::from(v));
    }
    fn elem_string(&mut self, s: &[u8]) {
        self.varint(s.len() as u64);
        self.buf.extend_from_slice(s);
    }
}

#[derive(Clone, Copy)]
enum Logical {
    None,
    Utf8,
    Date,
}

/// (name, physical type, logical type, optional)
const SCHEMA: [(&str, i32, Logical, bool); 6] = [
    ("user_id", 6, Logical::Utf8, false),
    ("day", 1, Logical::Date, false),
    ("sessions", 1, Logical::None, false),
    ("events", 1, Logical::None, false),
    ("revenue_cents", 1, Logical::None, false),
    ("top_category", 6, Logical::Utf8, true),
];

fn plain_bytes(v: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    out.extend_from_slice(v);
}

fn plain_i32(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// user_daily rows `(user_id, day, sessions, events, revenue, top_category)`
/// as one Parquet file in memory.
pub fn user_rows(rows: &[(String, i32, i32, i32, i32, String)]) -> Vec<u8> {
    let n = rows.len();
    let mut ids = Vec::new();
    let mut tops = Vec::new();
    for r in rows {
        plain_bytes(r.0.as_bytes(), &mut ids);
    }
    // Definition levels of the optional column: a 4-byte length, then one
    // RLE run of n ones (bit width 1).
    let mut lv = Vec::new();
    let mut v = (n as u64) << 1;
    while v >= 0x80 {
        lv.push((v as u8) | 0x80);
        v >>= 7;
    }
    lv.push(v as u8);
    lv.push(1);
    tops.extend_from_slice(&(lv.len() as u32).to_le_bytes());
    tops.extend_from_slice(&lv);
    for r in rows {
        plain_bytes(r.5.as_bytes(), &mut tops);
    }
    let cols: [Vec<u8>; 6] = [
        ids,
        plain_i32(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
        plain_i32(&rows.iter().map(|r| r.2).collect::<Vec<_>>()),
        plain_i32(&rows.iter().map(|r| r.3).collect::<Vec<_>>()),
        plain_i32(&rows.iter().map(|r| r.4).collect::<Vec<_>>()),
        tops,
    ];
    let mut file = b"PAR1".to_vec();
    let mut chunks = Vec::new();
    for body in &cols {
        let start = file.len() as i64;
        let mut t = Thrift::new();
        t.i32(1, 0);
        t.i32(2, body.len() as i32);
        t.i32(3, body.len() as i32);
        t.begin_struct(5);
        t.i32(1, n as i32);
        t.i32(2, ENC_PLAIN);
        t.i32(3, ENC_RLE);
        t.i32(4, ENC_RLE);
        t.end_struct();
        t.buf.push(0);
        file.extend_from_slice(&t.buf);
        file.extend_from_slice(body);
        chunks.push((start, file.len() as i64 - start));
    }
    let group_bytes = file.len() as i64 - 4;
    let mut t = Thrift::new();
    t.i32(1, 1);
    t.begin_list(2, T_STRUCT, SCHEMA.len() + 1);
    t.begin_elem();
    t.string(4, b"schema");
    t.i32(5, SCHEMA.len() as i32);
    t.end_struct();
    for (name, phys, logical, optional) in SCHEMA {
        t.begin_elem();
        t.i32(1, phys);
        t.i32(3, i32::from(optional));
        t.string(4, name.as_bytes());
        match logical {
            Logical::Utf8 => {
                t.i32(6, 0);
                t.begin_struct(10);
                t.begin_struct(1);
                t.end_struct();
                t.end_struct();
            }
            Logical::Date => {
                t.i32(6, 6);
                t.begin_struct(10);
                t.begin_struct(6);
                t.end_struct();
                t.end_struct();
            }
            Logical::None => {}
        }
        t.end_struct();
    }
    t.i64(3, n as i64);
    t.begin_list(4, T_STRUCT, 1);
    t.begin_elem();
    t.begin_list(1, T_STRUCT, SCHEMA.len());
    for (col, (start, size)) in chunks.iter().enumerate() {
        let (name, phys, _, _) = SCHEMA[col];
        t.begin_elem();
        t.i64(2, *start);
        t.begin_struct(3);
        t.i32(1, phys);
        t.begin_list(2, T_I32, 2);
        t.elem_i32(ENC_PLAIN);
        t.elem_i32(ENC_RLE);
        t.begin_list(3, T_BINARY, 1);
        t.elem_string(name.as_bytes());
        t.i32(4, 0);
        t.i64(5, n as i64);
        t.i64(6, *size);
        t.i64(7, *size);
        t.i64(9, *start);
        t.end_struct();
        t.end_struct();
    }
    t.i64(2, group_bytes);
    t.i64(3, n as i64);
    t.end_struct();
    t.string(6, b"sessions");
    t.buf.push(0);
    file.extend_from_slice(&t.buf);
    file.extend_from_slice(&(t.buf.len() as u32).to_le_bytes());
    file.extend_from_slice(b"PAR1");
    file
}
