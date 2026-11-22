//! A minimal Parquet writer: data page v1, uncompressed, PLAIN values or
//! RLE_DICTIONARY indices, flat schemas, no statistics. Only what the two
//! output tables need, so every byte written is accounted for here.

use std::ffi::CString;
use std::path::Path;

use crate::common::die;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phys {
    Int32 = 1,
    Int64 = 2,
    ByteArray = 6,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Logical {
    None,
    Utf8,
    Date,
}

#[derive(Clone, Copy)]
pub struct ColSpec {
    pub name: &'static str,
    pub phys: Phys,
    pub logical: Logical,
    pub optional: bool,
}

const ENC_PLAIN: i32 = 0;
const ENC_RLE: i32 = 3;
const ENC_RLE_DICT: i32 = 8;

/// Thrift compact protocol, write side only.
pub struct Thrift {
    pub buf: Vec<u8>,
    last: Vec<i16>,
    cur: i16,
}

const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_BINARY: u8 = 8;
const T_LIST: u8 = 9;
const T_STRUCT: u8 = 12;

impl Thrift {
    pub fn new() -> Self {
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
    pub fn i32(&mut self, id: i16, v: i32) {
        self.field(id, T_I32);
        self.zigzag(i64::from(v));
    }
    pub fn i64(&mut self, id: i16, v: i64) {
        self.field(id, T_I64);
        self.zigzag(v);
    }
    pub fn string(&mut self, id: i16, s: &[u8]) {
        self.field(id, T_BINARY);
        self.varint(s.len() as u64);
        self.buf.extend_from_slice(s);
    }
    pub fn begin_struct(&mut self, id: i16) {
        self.field(id, T_STRUCT);
        self.last.push(self.cur);
        self.cur = 0;
    }
    pub fn end_struct(&mut self) {
        self.buf.push(0);
        self.cur = self.last.pop().unwrap_or(0);
    }
    pub fn begin_list(&mut self, id: i16, elem: u8, len: usize) {
        self.field(id, T_LIST);
        self.list_header(elem, len);
    }
    fn list_header(&mut self, elem: u8, len: usize) {
        if len < 15 {
            self.buf.push(((len as u8) << 4) | elem);
        } else {
            self.buf.push(0xF0 | elem);
            self.varint(len as u64);
        }
    }
    /// A struct that is a list element: no field header.
    pub fn begin_elem(&mut self) {
        self.last.push(self.cur);
        self.cur = 0;
    }
    pub fn elem_i32(&mut self, v: i32) {
        self.zigzag(i64::from(v));
    }
    pub fn elem_string(&mut self, s: &[u8]) {
        self.varint(s.len() as u64);
        self.buf.extend_from_slice(s);
    }
}

/// One page of a column chunk: the header is built here, the body is the
/// concatenation of `parts`, borrowed from the caller's buffers.
pub struct Page<'a> {
    pub dict: bool,
    pub num_values: usize,
    pub encoding_dict: bool,
    pub parts: Vec<&'a [u8]>,
}

impl<'a> Page<'a> {
    pub fn plain(num_values: usize, parts: Vec<&'a [u8]>) -> Self {
        Page { dict: false, num_values, encoding_dict: false, parts }
    }
    pub fn dict_page(num_values: usize, parts: Vec<&'a [u8]>) -> Self {
        Page { dict: true, num_values, encoding_dict: false, parts }
    }
    pub fn dict_indices(num_values: usize, parts: Vec<&'a [u8]>) -> Self {
        Page { dict: false, num_values, encoding_dict: true, parts }
    }
}

struct ChunkMeta {
    first_offset: u64,
    data_offset: u64,
    dict_offset: Option<u64>,
    size: u64,
    num_values: u64,
    has_dict: bool,
}

struct GroupMeta {
    rows: u64,
    bytes: u64,
    chunks: Vec<ChunkMeta>,
}

#[cfg(target_os = "linux")]
fn early_writeback() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_WRITEBACK").map_or(true, |v| v != "0"))
}

pub struct PqFile {
    fd: i32,
    offset: u64,
    schema: Vec<ColSpec>,
    groups: Vec<GroupMeta>,
}

pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // The targets are little-endian; Parquet PLAIN is little-endian.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn write_all_vectored(fd: i32, parts: &[&[u8]]) {
    // One writev per batch of slices; loop on short writes.
    let mut iov: Vec<libc::iovec> = parts
        .iter()
        .filter(|p| !p.is_empty())
        .map(|p| libc::iovec { iov_base: p.as_ptr() as *mut libc::c_void, iov_len: p.len() })
        .collect();
    let mut at = 0;
    while at < iov.len() {
        let n = (iov.len() - at).min(512);
        let wrote = unsafe { libc::writev(fd, iov[at..].as_ptr(), n as i32) };
        if wrote < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            die(&format!("write: {err}"));
        }
        let mut left = wrote as usize;
        while at < iov.len() && left >= iov[at].iov_len {
            left -= iov[at].iov_len;
            at += 1;
        }
        if left > 0 {
            iov[at].iov_base = unsafe { (iov[at].iov_base as *mut u8).add(left) } as *mut libc::c_void;
            iov[at].iov_len -= left;
        }
    }
}

impl PqFile {
    pub fn create(path: &Path, schema: &[ColSpec]) -> Self {
        let c = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        let fd = unsafe {
            libc::open(c.as_ptr(), libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC, 0o666 as libc::c_uint)
        };
        if fd < 0 {
            die(&format!("{}: {}", path.display(), std::io::Error::last_os_error()));
        }
        write_all_vectored(fd, &[b"PAR1"]);
        PqFile { fd, offset: 4, schema: schema.to_vec(), groups: Vec::new() }
    }

    /// Write one row group: one column chunk per schema column, each a list
    /// of pages (an optional dictionary page first, then data pages).
    pub fn write_row_group(&mut self, rows: usize, columns: &[Vec<Page>]) {
        assert_eq!(columns.len(), self.schema.len());
        let mut headers: Vec<Vec<u8>> = Vec::new();
        let mut chunks = Vec::new();
        let mut offset = self.offset;
        for (col, pages) in columns.iter().enumerate() {
            let spec = self.schema[col];
            let first_offset = offset;
            let mut data_offset = 0;
            let mut dict_offset = None;
            let mut num_values = 0;
            for page in pages {
                let body: usize = page.parts.iter().map(|p| p.len()).sum();
                let mut t = Thrift::new();
                t.i32(1, if page.dict { 2 } else { 0 });
                t.i32(2, body as i32);
                t.i32(3, body as i32);
                if page.dict {
                    t.begin_struct(7);
                    t.i32(1, page.num_values as i32);
                    t.i32(2, ENC_PLAIN);
                    t.end_struct();
                    dict_offset = Some(offset);
                } else {
                    t.begin_struct(5);
                    t.i32(1, page.num_values as i32);
                    t.i32(2, if page.encoding_dict { ENC_RLE_DICT } else { ENC_PLAIN });
                    t.i32(3, ENC_RLE);
                    t.i32(4, ENC_RLE);
                    t.end_struct();
                    if data_offset == 0 {
                        data_offset = offset;
                    }
                    num_values += page.num_values as u64;
                }
                t.buf.push(0);
                offset += (t.buf.len() + body) as u64;
                headers.push(t.buf);
            }
            let _ = spec;
            chunks.push(ChunkMeta {
                first_offset,
                data_offset,
                dict_offset,
                size: offset - first_offset,
                num_values,
                has_dict: dict_offset.is_some(),
            });
        }
        let mut parts: Vec<&[u8]> = Vec::new();
        let mut h = 0;
        for pages in columns {
            for page in pages {
                parts.push(&headers[h]);
                h += 1;
                parts.extend_from_slice(&page.parts);
            }
        }
        write_all_vectored(self.fd, &parts);
        #[cfg(target_os = "linux")]
        if early_writeback() {
            // Ask the kernel to start writing these pages now, without
            // waiting for them. See NOTES.md, exit cost.
            unsafe {
                libc::sync_file_range(self.fd, self.offset as i64, (offset - self.offset) as i64, libc::SYNC_FILE_RANGE_WRITE);
            }
        }
        self.groups.push(GroupMeta { rows: rows as u64, bytes: offset - self.offset, chunks });
        self.offset = offset;
    }

    pub fn finish(self) {
        let mut t = Thrift::new();
        t.i32(1, 1);
        t.begin_list(2, T_STRUCT, self.schema.len() + 1);
        t.begin_elem();
        t.string(4, b"schema");
        t.i32(5, self.schema.len() as i32);
        t.end_struct();
        for spec in &self.schema {
            t.begin_elem();
            t.i32(1, spec.phys as i32);
            t.i32(3, i32::from(spec.optional));
            t.string(4, spec.name.as_bytes());
            match spec.logical {
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
        t.i64(3, self.groups.iter().map(|g| g.rows as i64).sum());
        t.begin_list(4, T_STRUCT, self.groups.len());
        for group in &self.groups {
            t.begin_elem();
            t.begin_list(1, T_STRUCT, group.chunks.len());
            for (col, chunk) in group.chunks.iter().enumerate() {
                let spec = self.schema[col];
                t.begin_elem();
                t.i64(2, chunk.first_offset as i64);
                t.begin_struct(3);
                t.i32(1, spec.phys as i32);
                if chunk.has_dict {
                    t.begin_list(2, T_I32, 3);
                    t.elem_i32(ENC_PLAIN);
                    t.elem_i32(ENC_RLE);
                    t.elem_i32(ENC_RLE_DICT);
                } else {
                    t.begin_list(2, T_I32, 2);
                    t.elem_i32(ENC_PLAIN);
                    t.elem_i32(ENC_RLE);
                }
                t.begin_list(3, T_BINARY, 1);
                t.elem_string(spec.name.as_bytes());
                t.i32(4, 0);
                t.i64(5, chunk.num_values as i64);
                t.i64(6, chunk.size as i64);
                t.i64(7, chunk.size as i64);
                t.i64(9, chunk.data_offset as i64);
                if let Some(d) = chunk.dict_offset {
                    t.i64(11, d as i64);
                }
                t.end_struct();
                t.end_struct();
            }
            t.i64(2, group.bytes as i64);
            t.i64(3, group.rows as i64);
            t.end_struct();
        }
        t.string(6, b"sessions");
        t.buf.push(0);
        let len = (t.buf.len() as u32).to_le_bytes();
        write_all_vectored(self.fd, &[&t.buf, &len, b"PAR1"]);
        unsafe { libc::close(self.fd) };
    }
}

/// The RLE / bit-packed hybrid, as one bit-packed run. `values` are already
/// smaller than `1 << width`. Used for dictionary indices (prefixed by the
/// width byte by the caller) and for definition levels.
pub fn bitpack_run(values: impl Iterator<Item = u32>, count: usize, width: u32, out: &mut Vec<u8>) {
    if count == 0 {
        return;
    }
    debug_assert!(width >= 1);
    let groups = count.div_ceil(8);
    let mut v = ((groups as u64) << 1) | 1;
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
    let start = out.len();
    let total = groups * width as usize;
    // Room for the packed bytes plus one 8-byte store past them. Values are
    // collected 32 bits at a time and stored without capacity checks.
    out.reserve(total + 8);
    debug_assert!(width <= 32);
    let mut acc: u64 = 0;
    let mut bits = 0u32;
    let mut n = 0usize;
    unsafe {
        let base = out.as_mut_ptr().add(start);
        let mut p = base;
        for value in values {
            debug_assert!(n < count && u64::from(value) >> width == 0);
            n += 1;
            acc |= u64::from(value) << bits;
            bits += width;
            if bits >= 32 {
                (p as *mut u32).write_unaligned((acc as u32).to_le());
                p = p.add(4);
                acc >>= 32;
                bits -= 32;
            }
        }
        // The rest of the bits, then zeros up to the whole groups. Bits of
        // `acc` above `bits` are zero, so the 8-byte store writes zeros.
        (p as *mut u64).write_unaligned(acc.to_le());
        let written = p.offset_from(base) as usize + 8;
        for i in written.min(total)..total {
            *base.add(i) = 0;
        }
        out.set_len(start + total);
    }
    debug_assert!(n <= count);
}

/// One RLE run of the hybrid encoding: `count` copies of `value`.
pub fn rle_run(count: usize, value: u32, width: u32, out: &mut Vec<u8>) {
    let mut v = (count as u64) << 1;
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
    out.extend_from_slice(&value.to_le_bytes()[..width.div_ceil(8) as usize]);
}

/// Definition levels for an optional flat column: a 4-byte length, then the
/// hybrid run of 0/1 levels.
pub fn def_levels(present: impl Iterator<Item = bool>, count: usize, out: &mut Vec<u8>) {
    let at = out.len();
    out.extend_from_slice(&[0; 4]);
    bitpack_run(present.map(u32::from), count, 1, out);
    let len = (out.len() - at - 4) as u32;
    out[at..at + 4].copy_from_slice(&len.to_le_bytes());
}

pub fn plain_bytes(value: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value);
}
