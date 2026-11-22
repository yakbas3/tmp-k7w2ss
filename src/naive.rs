//! The reference pipeline: single-threaded, hash maps and sorts, written
//! straight from rules.md. It is the oracle for the tests and the fallback
//! the binary uses when a dataset breaks a precondition of the fast engine.

use std::collections::HashMap;
use std::path::Path;

use crate::common::*;

struct Event<'a> {
    user: &'a [u8],
    ts: &'a [u8],
    ingest: i64,
    kind: &'a [u8],
    product: &'a [u8],
    price: &'a [u8],
}

struct Click<'a> {
    id: &'a [u8],
    ts: i64,
    camp: u32,
}

struct Kept {
    ts: i64,
    price: u64,
    cat: u32,
    purchase: bool,
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let up = parent[parent[x as usize] as usize];
        parent[x as usize] = up;
        x = up;
    }
    x
}

pub fn run(data: &Path, lk: &Lookups) -> Tables {
    let read_all = |sub: &str| -> Vec<Vec<u8>> {
        list_csv(&data.join(sub))
            .iter()
            .map(|p| std::fs::read(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))))
            .collect()
    };
    let event_files = read_all("events");
    let click_files = read_all("clicks");
    let link_files = read_all("links");
    let mut fields: Vec<&[u8]> = Vec::new();

    // Linked IDs: union-find over exact byte strings.
    let mut index: HashMap<&[u8], u32> = HashMap::new();
    let mut names: Vec<&[u8]> = Vec::new();
    let mut parent: Vec<u32> = Vec::new();
    for file in &link_files {
        for line in data_lines(file) {
            if !split_line(line, 2, &mut fields) || fields[0].is_empty() || fields[1].is_empty() {
                continue;
            }
            let mut ids = [0u32; 2];
            for side in 0..2 {
                let next = names.len() as u32;
                let id = *index.entry(fields[side]).or_insert(next);
                if id == next {
                    names.push(fields[side]);
                    parent.push(next);
                }
                ids[side] = id;
            }
            let (a, b) = (find(&mut parent, ids[0]), find(&mut parent, ids[1]));
            if a != b {
                parent[a as usize] = b;
            }
        }
    }
    let mut best: Vec<&[u8]> = names.clone();
    for i in 0..names.len() as u32 {
        let root = find(&mut parent, i) as usize;
        if names[i as usize] < best[root] {
            best[root] = names[i as usize];
        }
    }
    let person_of = |id: &[u8], parent: &mut Vec<u32>| -> Vec<u8> {
        match index.get(id) {
            Some(&i) => best[find(parent, i) as usize].to_vec(),
            None => id.to_vec(),
        }
    };

    // Events: parse, dedup on event_id keeping the smallest ingest_ts.
    let mut kept: HashMap<&[u8], Event> = HashMap::new();
    for file in &event_files {
        for line in data_lines(file) {
            if !split_line(line, 7, &mut fields) || fields[0].is_empty() {
                continue;
            }
            // ingest_ts is always valid F1 by the rules. If it is not, the
            // copy sorts last and can never be a valid kept copy.
            let ingest = parse_ts_naive(fields[3]);
            let event = Event {
                user: fields[1],
                ts: if ingest.is_some() { fields[2] } else { b"" },
                ingest: ingest.unwrap_or(i64::MAX),
                kind: fields[4],
                product: fields[5],
                price: fields[6],
            };
            match kept.get_mut(fields[0]) {
                Some(old) => {
                    if event.ingest < old.ingest {
                        *old = event;
                    }
                }
                None => {
                    kept.insert(fields[0], event);
                }
            }
        }
    }

    // Validate, drop late, hand each event to its person.
    let window = lk.window_start..lk.window_end;
    let mut people: HashMap<Vec<u8>, (Vec<Kept>, Vec<Click>)> = HashMap::new();
    for event in kept.values() {
        if event.user.is_empty() || event.product.is_empty() {
            continue;
        }
        let Some(ts) = parse_ts_naive(event.ts) else {
            continue;
        };
        if !window.contains(&ts) {
            continue;
        }
        let purchase = match event.kind {
            b"view" | b"cart" => false,
            b"purchase" => true,
            _ => continue,
        };
        let mut price = 0;
        if purchase {
            match parse_price_naive(event.price) {
                Some(p) => price = p,
                None => continue,
            }
        }
        if event.ingest.saturating_sub(ts) > LATE_MS {
            continue;
        }
        let cat = lk.product_cat(event.product);
        people
            .entry(person_of(event.user, &mut parent))
            .or_default()
            .0
            .push(Kept { ts, price, cat, purchase });
    }
    drop(kept);

    // Clicks. Copies of a click are identical, so a duplicate can change
    // neither the credited click nor the set of assisting campaigns; the
    // dedup is still done here, as the rules state it.
    let click_window = lk.window_start - CLICK_REACH_MS..lk.window_end;
    let mut seen: HashMap<&[u8], ()> = HashMap::new();
    for file in &click_files {
        for line in data_lines(file) {
            if !split_line(line, 4, &mut fields) || fields[0].is_empty() {
                continue;
            }
            if seen.insert(fields[0], ()).is_some() {
                continue;
            }
            if fields[1].is_empty() {
                continue;
            }
            let Some(ts) = parse_ts_naive(fields[3]) else {
                continue;
            };
            if !click_window.contains(&ts) {
                continue;
            }
            let Some(&camp) = lk.campaigns.get(fields[2]) else {
                continue;
            };
            if let Some(person) = people.get_mut(&person_of(fields[1], &mut parent)) {
                person.1.push(Click { id: fields[0], ts, camp });
            }
        }
    }

    let organic = lk.camp_ids.len() as u32;
    // (campaign, day) -> [sessions, purchasing, revenue, assisted]
    let mut camp: HashMap<(u32, i32), [i64; 4]> = HashMap::new();
    let mut tables = Tables::default();
    for (name, (mut events, mut clicks)) in people {
        events.sort_by_key(|e| e.ts);
        clicks.sort_by(|a, b| a.ts.cmp(&b.ts).then(b.id.cmp(a.id)));
        // day -> (sessions, events, revenue, category -> revenue)
        let mut days: Vec<(i32, i64, i64, i64, HashMap<u32, i64>)> = Vec::new();
        let mut at = 0;
        while at < events.len() {
            let start = events[at].ts;
            let mut end = at + 1;
            while end < events.len() && events[end].ts - events[end - 1].ts <= SESSION_GAP_MS {
                end += 1;
            }
            let day = start.div_euclid(DAY_MS) as i32;
            if days.last().map(|d| d.0) != Some(day) {
                days.push((day, 0, 0, 0, HashMap::new()));
            }
            let row = days.last_mut().unwrap();
            row.1 += 1;
            row.2 += (end - at) as i64;
            let mut revenue = 0;
            let mut purchased = false;
            for e in &events[at..end] {
                if e.purchase {
                    purchased = true;
                    revenue += e.price as i64;
                    *row.4.entry(e.cat).or_default() += e.price as i64;
                }
            }
            row.3 += revenue;

            // Clicks are sorted by ts, then by id descending, so the last
            // one at or before the start is the latest with the smallest id.
            let qualifying: Vec<&Click> = clicks
                .iter()
                .filter(|c| c.ts <= start && start - c.ts <= CLICK_REACH_MS)
                .collect();
            let credited = qualifying.last().map_or(organic, |c| c.camp);
            let cell = camp.entry((credited, day)).or_default();
            cell[0] += 1;
            cell[1] += i64::from(purchased);
            cell[2] += revenue;
            let mut assists: Vec<u32> = qualifying.iter().map(|c| c.camp).collect();
            assists.sort_unstable();
            assists.dedup();
            for c in assists {
                camp.entry((c, day)).or_default()[3] += 1;
            }
            at = end;
        }
        for (day, sessions, count, revenue, cats) in days {
            let top = cats
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(&c, _)| lk.cats[c as usize].clone());
            tables.user.push(UserRow {
                user_id: name.clone(),
                day,
                sessions,
                events: count,
                revenue,
                top,
            });
        }
    }
    for ((c, day), cell) in camp {
        let (campaign, channel) = if c == organic {
            (b"organic".to_vec(), b"organic".to_vec())
        } else {
            (lk.camp_ids[c as usize].clone(), lk.camp_channels[c as usize].clone())
        };
        tables.camp.push(CampRow {
            campaign,
            channel,
            day,
            sessions: cell[0],
            purchasing: cell[1],
            revenue: cell[2],
            assisted: cell[3],
        });
    }
    tables.sort();
    tables
}
