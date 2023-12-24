use crate::{NodeIdx, VertIdx, NULL};
use std::ops::Sub;

#[derive(Clone, Debug)]
pub struct Node {
    pub i: VertIdx,         // vertex index in flat one-d array of 64bit float coords
    pub x: f64,             // vertex x coordinate
    pub y: f64,             // vertex y coordinate
    pub prev_idx: NodeIdx,  // previous vertex node in a polygon ring
    pub next_idx: NodeIdx,  // next vertex node in a polygon ring
    pub z: i32,             // z-order curve value
    pub prevz_idx: NodeIdx, // previous node in z-order
    pub nextz_idx: NodeIdx, // next node in z-order
    pub steiner: bool,      // indicates whether this is a steiner point
    pub idx: NodeIdx,       // index within LinkedLists vector that holds all nodes
}

impl<'a> Sub for &'a Node {
    type Output = Node;

    fn sub(self, other: &'a Node) -> Node {
        Node::new(NULL, self.x - other.x, self.y - other.y, NULL)
    }
}

// check if two points are equal
pub fn equal_coords(p1: &Node, p2: &Node) -> bool {
    p1.x == p2.x && p1.y == p2.y
}

impl Node {
    pub fn new(i: VertIdx, x: f64, y: f64, idx: NodeIdx) -> Node {
        Node {
            i: i,
            x: x,
            y: y,
            prev_idx: NULL,
            next_idx: NULL,
            z: 0,
            nextz_idx: NULL,
            prevz_idx: NULL,
            steiner: false,
            idx: idx,
        }
    }
    pub fn prev<'a>(&'a self, ll: &'a LinkedLists) -> &Node {
        &self.idx.prev(ll).node(ll)
    }
    pub fn next<'a>(&'a self, ll: &'a LinkedLists) -> &Node {
        &self.idx.next(ll).node(ll)
    }
}

// Extension Trait for NodeIdx
// this allows us to do stuff like i.next(ll).next(ll)
// which helps to write linked list code in a more concise, clear fashion
pub trait NodeIndex {
    fn next(self, ll: &LinkedLists) -> NodeIdx;
    fn nextz(self, ll: &LinkedLists) -> NodeIdx;
    fn prevz(self, ll: &LinkedLists) -> NodeIdx;
    fn prev(self, ll: &LinkedLists) -> NodeIdx;
    fn set_next(self, ll: &mut LinkedLists, i: NodeIdx);
    fn set_nextz(self, ll: &mut LinkedLists, i: NodeIdx);
    fn node(self, ll: &LinkedLists) -> &Node;
    fn set_prev(self, ll: &mut LinkedLists, i: NodeIdx);
    fn set_prevz(self, ll: &mut LinkedLists, i: NodeIdx);
    fn set_z(self, ll: &mut LinkedLists, z: i32);
    fn set_steiner(self, ll: &mut LinkedLists, v: bool);
}

// Extension Trait implementation. NodeIdx is typically just usize
// so basically we are extending the integers to allow looking up nodes in a linked list
// by calling methods on integers. for example
//
//      i is an NodeIdx    ll is a LinkedLists
//      i.next(ll).next(ll)  // this gets the index of the node after the node after i
//
impl NodeIndex for NodeIdx {
    fn next(self, ll: &LinkedLists) -> NodeIdx {
        ll.nodes[self].next_idx
    }
    fn nextz(self, ll: &LinkedLists) -> NodeIdx {
        ll.nodes[self].nextz_idx
    }
    fn prevz(self, ll: &LinkedLists) -> NodeIdx {
        ll.nodes[self].prevz_idx
    }
    fn prev(self, ll: &LinkedLists) -> NodeIdx {
        ll.nodes[self].prev_idx
    }
    fn set_next(self, ll: &mut LinkedLists, i: NodeIdx) {
        ll.nodes.get_mut(self).unwrap().next_idx = i;
    }
    fn set_nextz(self, ll: &mut LinkedLists, i: NodeIdx) {
        ll.nodes.get_mut(self).unwrap().nextz_idx = i;
    }
    fn set_prev(self, ll: &mut LinkedLists, i: NodeIdx) {
        ll.nodes.get_mut(self).unwrap().prev_idx = i;
    }
    fn set_prevz(self, ll: &mut LinkedLists, i: NodeIdx) {
        ll.nodes.get_mut(self).unwrap().prevz_idx = i;
    }
    fn set_z(self, ll: &mut LinkedLists, z: i32) {
        ll.nodes.get_mut(self).unwrap().z = z;
    }
    fn set_steiner(self, ll: &mut LinkedLists, v: bool) {
        ll.nodes.get_mut(self).unwrap().steiner = v;
    }
    fn node(self, ll: &LinkedLists) -> &Node {
        &ll.nodes[self]
    }
}

pub struct LinkedLists {
    pub nodes: Vec<Node>,
    pub invsize: f64,
    pub minx: f64,
    pub miny: f64,
    pub maxx: f64,
    pub maxy: f64,
}

#[macro_export]
macro_rules! dlog {
	($loglevel:expr, $($s:expr),*) => (
		if DEBUG>=$loglevel { print!("{}:",$loglevel); println!($($s),+); }
	)
}

impl LinkedLists {
    pub fn iter(&self, r: std::ops::Range<NodeIdx>) -> NodeIterator {
        return NodeIterator::new(self, r.start, r.end);
    }
    pub fn iter_pairs(&self, r: std::ops::Range<NodeIdx>) -> NodePairIterator {
        return NodePairIterator::new(self, r.start, r.end);
    }
    pub fn insert_node(&mut self, i: VertIdx, x: f64, y: f64, last: NodeIdx) -> NodeIdx {
        let mut p = Node::new(i, x, y, self.nodes.len());
        if last == NULL {
            p.next_idx = p.idx;
            p.prev_idx = p.idx;
        } else {
            p.next_idx = last.next(self);
            p.prev_idx = last;
            let lastnextidx = last.next(self);
            lastnextidx.set_prev(self, p.idx);
            last.set_next(self, p.idx);
        };
        let result = p.idx;
        self.nodes.push(p);
        return result;
    }
    pub fn remove_node(&mut self, p_idx: NodeIdx) {
        let pi = p_idx.prev(self);
        let ni = p_idx.next(self);
        let pz = p_idx.prevz(self);
        let nz = p_idx.nextz(self);
        pi.set_next(self, ni);
        ni.set_prev(self, pi);
        pz.set_nextz(self, nz);
        nz.set_prevz(self, pz);
    }
    pub fn new(size_hint: usize) -> LinkedLists {
        let mut ll = LinkedLists {
            nodes: Vec::with_capacity(size_hint),
            invsize: 0.0,
            minx: std::f64::MAX,
            miny: std::f64::MAX,
            maxx: std::f64::MIN,
            maxy: std::f64::MIN,
        };
        // ll.nodes[0] is the NULL node. For example usage, see remove_node()
        ll.nodes.push(Node {
            i: 0,
            x: 0.0,
            y: 0.0,
            prev_idx: 0,
            next_idx: 0,
            z: 0,
            nextz_idx: 0,
            prevz_idx: 0,
            steiner: false,
            idx: 0,
        });
        ll
    }
    pub fn node(&self, i: NodeIdx) -> &Node {
        &self.nodes[i]
    }
}

pub struct NodeIterator<'a> {
    cur: NodeIdx,
    end: NodeIdx,
    ll: &'a LinkedLists,
    pending_result: Option<&'a Node>,
}

impl<'a> NodeIterator<'a> {
    pub fn new(ll: &LinkedLists, start: NodeIdx, end: NodeIdx) -> NodeIterator {
        NodeIterator {
            pending_result: Some(ll.node(start)),
            cur: start,
            end: end,
            ll,
        }
    }
}

impl<'a> Iterator for NodeIterator<'a> {
    type Item = &'a Node;
    fn next(&mut self) -> Option<Self::Item> {
        self.cur = self.cur.next(self.ll);
        let cur_result = self.pending_result;
        if self.cur == self.end {
            // only one branch, saves time
            self.pending_result = None;
        } else {
            self.pending_result = Some(self.ll.node(self.cur));
        }
        cur_result
    }
}

pub struct NodePairIterator<'a> {
    cur: NodeIdx,
    end: NodeIdx,
    ll: &'a LinkedLists,
    pending_result: Option<(&'a Node, &'a Node)>,
}

impl<'a> NodePairIterator<'a> {
    pub fn new(ll: &LinkedLists, start: NodeIdx, end: NodeIdx) -> NodePairIterator {
        NodePairIterator {
            pending_result: Some((ll.node(start), start.next(ll).node(ll))),
            cur: start,
            end: end,
            ll,
        }
    }
}

impl<'a> Iterator for NodePairIterator<'a> {
    type Item = (&'a Node, &'a Node);
    fn next(&mut self) -> Option<Self::Item> {
        self.cur = self.cur.next(self.ll);
        let cur_result = self.pending_result;
        if self.cur == self.end {
            // only one branch, saves time
            self.pending_result = None;
        } else {
            self.pending_result =
                Some((self.ll.node(self.cur), self.ll.node(self.cur).next(self.ll)))
        }
        cur_result
    }
}

pub fn pn(a: usize) -> String {
    match a {
        0x777A91CC => String::from("NULL"),
        _ => a.to_string(),
    }
}
pub fn pb(a: bool) -> String {
    match a {
        true => String::from("x"),
        false => String::from(" "),
    }
}
pub fn dump(ll: &LinkedLists) -> String {
    let mut s = format!("LL, #nodes: {}", ll.nodes.len());
    s.push_str(&format!(
        " #used: {}\n",
        //        ll.nodes.len() as i64 - ll.freelist.len() as i64
        ll.nodes.len() as i64
    ));
    s.push_str(&format!(
        " {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2} {:>4}\n",
        "vi", "i", "p", "n", "x", "y", "pz", "nz", "st", "fr", "cyl", "z"
    ));
    for n in &ll.nodes {
        s.push_str(&format!(
            " {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2} {:>4}\n",
            n.idx,
            n.i,
            pn(n.prev_idx),
            pn(n.next_idx),
            n.x,
            n.y,
            pn(n.prevz_idx),
            pn(n.nextz_idx),
            pb(n.steiner),
            false,
            //            pb(ll.freelist.contains(&n.idx)),
            0, //,ll.iter(n.idx..n.idx).count(),
            n.z,
        ));
    }
    return s;
}

pub fn cycle_dump(ll: &LinkedLists, p: NodeIdx) -> String {
    let mut s = format!("cycle from {}, ", p);
    s.push_str(&format!(" len {}, idxs:", 0)); //cycle_len(&ll, p)));
    let mut i = p;
    let end = i;
    let mut count = 0;
    loop {
        count += 1;
        s.push_str(&format!("{} ", i.node(ll).idx));
        s.push_str(&format!("(i:{}), ", i.node(ll).i));
        i = i.next(ll);
        if i == end {
            break s;
        }
        if count > ll.nodes.len() {
            s.push_str(&format!(" infinite loop"));
            break s;
        }
    }
}

pub fn cycles_report(ll: &LinkedLists) -> String {
    if ll.nodes.len() == 1 {
        return format!("[]");
    }
    let mut markv: Vec<usize> = Vec::new();
    markv.resize(ll.nodes.len(), NULL);
    let mut cycler;
    for i in 0..markv.len() {
        //            if ll.freelist.contains(&i) {
        if true {
            markv[i] = NULL;
        } else if markv[i] == NULL {
            cycler = i;
            let mut p = i;
            let end = p.prev(ll);
            markv[p] = cycler;
            let mut count = 0;
            loop {
                p = p.next(ll);
                markv[p] = cycler;
                count += 1;
                if p == end || count > ll.nodes.len() {
                    break;
                }
            } // loop
        } // if markvi == 0
    } //for markv
    format!("cycles report:\n{:?}", markv)
}

pub fn dump_cycle(ll: &LinkedLists, start: usize) -> String {
    let mut s = format!("LL, #nodes: {}", ll.nodes.len());
    //        s.push_str(&format!(" #used: {}\n", ll.nodes.len() - ll.freelist.len()));
    s.push_str(&format!(" #used: {}\n", ll.nodes.len()));
    s.push_str(&format!(
        " {:>3} {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2}\n",
        "#", "vi", "i", "p", "n", "x", "y", "pz", "nz", "st", "fr", "cyl"
    ));
    let mut startidx: usize = 0;
    for n in &ll.nodes {
        if n.i == start {
            startidx = n.idx;
        };
    }
    let endidx = startidx;
    let mut idx = startidx;
    let mut count = 0;
    let mut state; // = 0i32;
    loop {
        let n = idx.node(ll);
        state = 0; //horsh( state, n.i  as i32);
        s.push_str(&format!(
            " {:>3} {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2}\n",
            count,
            n.idx,
            n.i,
            n.prev(ll).i,
            n.next(ll).i,
            n.x,
            n.y,
            pn(n.prevz_idx),
            pn(n.nextz_idx),
            pb(n.steiner),
            //                pb(ll.freelist.contains(&n.idx)),
            false,
            cycle_len(&ll, n.idx),
        ));
        idx = idx.next(ll);
        count += 1;
        if idx == endidx || count > ll.nodes.len() {
            break;
        }
    }
    s.push_str(&format!("dump end, horshcount:{} horsh:{}", count, state));
    return s;
}

pub fn cycle_len(ll: &LinkedLists, p: NodeIdx) -> usize {
    if p >= ll.nodes.len() {
        return 0;
    }
    let end = p.prev(ll);
    let mut i = p;
    let mut count = 1;
    loop {
        i = i.next(ll);
        count += 1;
        if i == end {
            break count;
        }
        if count > ll.nodes.len() {
            break count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // https://www.cs.hmc.edu/~geoff/classes/hmc.cs070.200101/homework10/hashfuncs.$
    // https://stackoverflow.com/questions/1908492/unsigned-integer-in-javascript
    fn horsh(mut h: u32, n: u32) -> u32 {
        let highorder = h & 0xf8000000; // extract high-order 5 bits from h
                                        // 0xf8000000 is the hexadecimal representat$
                                        //   for the 32-bit number with the first fi$
                                        //   bits = 1 and the other bits = 0
        h = h << 5; // shift h left by 5 bits
        h = h ^ (highorder >> 27); // move the highorder 5 bits to the low-ord$
                                   //   end and XOR into h
        h = h ^ n; // XOR h and ki
        return h;
    }

    // find the node with 'i' of starti, horsh it
    fn horsh_ll(ll: &LinkedLists, starti: VertIdx) -> String {
        let mut s = format!("LL horsh: ");
        let mut startidx: usize = 0;
        for n in &ll.nodes {
            if n.i == starti {
                startidx = n.idx;
            };
        }
        let endidx = startidx;
        let mut idx = startidx;
        let mut count = 0;
        let mut state = 0u32;
        loop {
            let n = idx.node(ll);
            state = horsh(state, n.i as u32);
            idx = idx.next(ll);
            count += 1;
            if idx == endidx || count > ll.nodes.len() {
                break;
            }
        }
        s.push_str(&format!(" count:{} horsh: {}", count, state));
        return s;
    }
}
