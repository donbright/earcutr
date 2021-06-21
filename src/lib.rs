#![allow(dead_code)]

use num_traits::float::Float;
use std::fmt::Display;

static DIM: usize = 2;
static NULL: usize = 0;
//static DEBUG: usize = 4;
static DEBUG: usize = 0; // dlogs get optimized away at 0

type NodeIdx = usize;
type VertIdx = usize;

#[derive(Clone, Debug)]
struct Node<T: Float + Display> {
    i: VertIdx,         // vertex index in flat one-d array of 64bit float coords
    x: T,               // vertex x coordinate
    y: T,               // vertex y coordinate
    prev_idx: NodeIdx,  // previous vertex node in a polygon ring
    next_idx: NodeIdx,  // next vertex node in a polygon ring
    z: i32,             // z-order curve value
    prevz_idx: NodeIdx, // previous node in z-order
    nextz_idx: NodeIdx, // next node in z-order
    steiner: bool,      // indicates whether this is a steiner point
    idx: NodeIdx,       // index within LinkedLists vector that holds all nodes
}

impl<T: Float + Display> Node<T> {
    fn new(i: VertIdx, x: T, y: T, idx: NodeIdx) -> Node<T> {
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
}

pub struct LinkedLists<T: Float + Display> {
    nodes: Vec<Node<T>>,
    invsize: T,
    minx: T,
    miny: T,
    maxx: T,
    maxy: T,
    usehash: bool,
}

macro_rules! dlog {
	($loglevel:expr, $($s:expr),*) => (
		if DEBUG>=$loglevel { print!("{}:",$loglevel); println!($($s),+); }
	)
}

// macro design: built so we can easily swap unchecked for checked,
// to test speed. and because unsafe get_ funcs have different meaning
// than bracket operator (indexing operator) nodes[index]
macro_rules! noderef {
    ($ll:expr,$idx:expr) => {
        unsafe { $ll.nodes.get_unchecked($idx) }
        //&$ll.nodes[$idx]
    };
}
macro_rules! node {
    ($ll:expr,$idx:expr) => {
        unsafe { $ll.nodes.get_unchecked($idx) }
        //$ll.nodes[$idx]
    };
}
macro_rules! nodemut {
    ($ll:expr,$idx:expr) => {
        unsafe { $ll.nodes.get_unchecked_mut($idx) }
        //$ll.nodes.get_mut($idx).unwrap()
    };
}
// Note: none of the following macros work for Left-Hand-Side of assignment.
macro_rules! next {
    ($ll:expr,$idx:expr) => {
        unsafe {
            $ll.nodes
                .get_unchecked($ll.nodes.get_unchecked($idx).next_idx)
        }
        //$ll.nodes[$ll.nodes[$idx].next_idx]
    };
}
macro_rules! nextref {
    ($ll:expr,$idx:expr) => {
        unsafe {
            &$ll.nodes
                .get_unchecked($ll.nodes.get_unchecked($idx).next_idx)
        }
        //&$ll.nodes[$ll.nodes[$idx].next_idx]
    };
}
macro_rules! prev {
    ($ll:expr,$idx:expr) => {
        unsafe {
            $ll.nodes
                .get_unchecked($ll.nodes.get_unchecked($idx).prev_idx)
        }
        //$ll.nodes[$ll.nodes[$idx].prev_idx]
    };
}
macro_rules! prevref {
    ($ll:expr,$idx:expr) => {
        unsafe {
            &$ll.nodes
                .get_unchecked($ll.nodes.get_unchecked($idx).prev_idx)
        }
        //&$ll.nodes[$ll.nodes[$idx].prev_idx]
    };
}
macro_rules! prevz {
    ($ll:expr,$idx:expr) => {
        &$ll.nodes[$ll.nodes[$idx].prevz_idx]
        /*unsafe {
            $ll.nodes
                .get_unchecked($ll.nodes.get_unchecked($idx).prevz_idx)
        }*/
    };
}

impl<T: Float + Display> LinkedLists<T> {
    fn iter(&self, r: std::ops::Range<NodeIdx>) -> NodeIterator<T> {
        return NodeIterator::new(self, r.start, r.end);
    }
    fn iter_pairs(&self, r: std::ops::Range<NodeIdx>) -> NodePairIterator<T> {
        return NodePairIterator::new(self, r.start, r.end);
    }
    fn insert_node(&mut self, i: VertIdx, x: T, y: T, last: NodeIdx) -> NodeIdx {
        let mut p = Node::new(i, x, y, self.nodes.len());
        if last == NULL {
            p.next_idx = p.idx;
            p.prev_idx = p.idx;
        } else {
            p.next_idx = noderef!(self, last).next_idx;
            p.prev_idx = last;
            let lastnextidx = noderef!(self, last).next_idx;
            nodemut!(self, lastnextidx).prev_idx = p.idx;
            nodemut!(self, last).next_idx = p.idx;
        };
        let result = p.idx;
        self.nodes.push(p);
        return result;
    }
    fn remove_node(&mut self, p_idx: NodeIdx) {
        let pi = noderef!(self, p_idx).prev_idx;
        let ni = noderef!(self, p_idx).next_idx;
        let pz = noderef!(self, p_idx).prevz_idx;
        let nz = noderef!(self, p_idx).nextz_idx;
        nodemut!(self, pi).next_idx = ni;
        nodemut!(self, ni).prev_idx = pi;
        nodemut!(self, pz).nextz_idx = nz;
        nodemut!(self, nz).prevz_idx = pz;
    }
    fn new(size_hint: usize) -> LinkedLists<T> {
        let mut ll = LinkedLists {
            nodes: Vec::with_capacity(size_hint),
            invsize: T::zero(),
            minx: T::max_value(),
            miny: T::max_value(),
            maxx: T::min_value(),
            maxy: T::min_value(),
            usehash: true,
        };
        // ll.nodes[0] is the NULL node. For example usage, see remove_node()
        ll.nodes.push(Node {
            i: 0,
            x: T::zero(),
            y: T::zero(),
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
}

struct NodeIterator<'a, T: Float + Display> {
    cur: NodeIdx,
    end: NodeIdx,
    ll: &'a LinkedLists<T>,
    pending_result: Option<&'a Node<T>>,
}

impl<'a, T: Float + Display> NodeIterator<'a, T> {
    fn new(ll: &LinkedLists<T>, start: NodeIdx, end: NodeIdx) -> NodeIterator<T> {
        NodeIterator {
            pending_result: Some(noderef!(ll, start)),
            cur: start,
            end: end,
            ll,
        }
    }
}

impl<'a, T: Float + Display> Iterator for NodeIterator<'a, T> {
    type Item = &'a Node<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.cur = noderef!(self.ll, self.cur).next_idx;
        let cur_result = self.pending_result;
        if self.cur == self.end {
            // only one branch, saves time
            self.pending_result = None;
        } else {
            self.pending_result = Some(noderef!(self.ll, self.cur));
        }
        cur_result
    }
}

struct NodePairIterator<'a, T: Float + Display> {
    cur: NodeIdx,
    end: NodeIdx,
    ll: &'a LinkedLists<T>,
    pending_result: Option<(&'a Node<T>, &'a Node<T>)>,
}

impl<'a, T: Float + Display> NodePairIterator<'a, T> {
    fn new(ll: &LinkedLists<T>, start: NodeIdx, end: NodeIdx) -> NodePairIterator<T> {
        NodePairIterator {
            pending_result: Some((noderef!(ll, start), nextref!(ll, start))),
            cur: start,
            end: end,
            ll,
        }
    }
}

impl<'a, T: Float + Display> Iterator for NodePairIterator<'a, T> {
    type Item = (&'a Node<T>, &'a Node<T>);
    fn next(&mut self) -> Option<Self::Item> {
        self.cur = node!(self.ll, self.cur).next_idx;
        let cur_result = self.pending_result;
        if self.cur == self.end {
            // only one branch, saves time
            self.pending_result = None;
        } else {
            self.pending_result = Some((noderef!(self.ll, self.cur), nextref!(self.ll, self.cur)))
        }
        cur_result
    }
}

fn compare_x<T: Float + Display>(a: &Node<T>, b: &Node<T>) -> std::cmp::Ordering {
    a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
}

// link every hole into the outer loop, producing a single-ring polygon
// without holes
fn eliminate_holes<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    data: &Vec<T>,
    hole_indices: &Vec<usize>,
    inouter_node: NodeIdx,
) -> NodeIdx {
    let mut outer_node = inouter_node;
    let mut queue: Vec<Node<T>> = Vec::new();
    for i in 0..hole_indices.len() {
        let start = hole_indices[i] * DIM;
        let end = if i < (hole_indices.len() - 1) {
            hole_indices[i + 1] * DIM
        } else {
            data.len()
        };
        let (list, leftmost_idx) = linked_list_add_contour(ll, &data, start, end, false);
        if list == noderef!(ll, list).next_idx {
            nodemut!(ll, list).steiner = true;
        }
        queue.push(node!(ll, leftmost_idx).clone());
    }

    queue.sort_by(compare_x);

    // process holes from left to right
    for i in 0..queue.len() {
        eliminate_hole(ll, queue[i].idx, outer_node);
        let nextidx = next!(ll, outer_node).idx;
        outer_node = filter_points(ll, outer_node, nextidx);
    }
    outer_node
} // elim holes

// minx, miny and invsize are later used to transform coords
// into integers for z-order calculation
fn calc_invsize<T: Float + Display>(minx: T, miny: T, maxx: T, maxy: T) -> T {
    let invsize = T::max(maxx - minx, maxy - miny);
    let zero = T::zero();
    match invsize == zero {
        true => zero,
        false => num_traits::cast::<f64, T>(32767.0).unwrap() / invsize,
    }
}

// main ear slicing loop which triangulates a polygon (given as a linked
// list)
fn earcut_linked_hashed<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    mut ear_idx: NodeIdx,
    triangles: &mut Vec<usize>,
    pass: usize,
) {
    // interlink polygon nodes in z-order
    if pass == 0 {
        index_curve(ll, ear_idx);
    }
    // iterate through ears, slicing them one by one
    let mut stop_idx = ear_idx;
    let mut prev_idx = 0;
    let mut next_idx = node!(ll, ear_idx).next_idx;
    while stop_idx != next_idx {
        prev_idx = node!(ll, ear_idx).prev_idx;
        next_idx = node!(ll, ear_idx).next_idx;
        if is_ear_hashed(ll, prev_idx, ear_idx, next_idx) {
            triangles.push(node!(ll, prev_idx).i);
            triangles.push(node!(ll, ear_idx).i);
            triangles.push(node!(ll, next_idx).i);
            ll.remove_node(ear_idx);
            // skipping the next vertex leads to less sliver triangles
            ear_idx = node!(ll, next_idx).next_idx;
            stop_idx = ear_idx;
        } else {
            ear_idx = next_idx;
        }
    }

    if prev_idx == next_idx {
        return;
    };
    // if we looped through the whole remaining polygon and can't
    // find any more ears
    if pass == 0 {
        let tmp = filter_points(ll, next_idx, NULL);
        earcut_linked_hashed(ll, tmp, triangles, 1);
    } else if pass == 1 {
        ear_idx = cure_local_intersections(ll, next_idx, triangles);
        earcut_linked_hashed(ll, ear_idx, triangles, 2);
    } else if pass == 2 {
        split_earcut(ll, next_idx, triangles);
    }
}

// main ear slicing loop which triangulates a polygon (given as a linked
// list)
fn earcut_linked_unhashed<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    mut ear_idx: NodeIdx,
    triangles: &mut Vec<usize>,
    pass: usize,
) {
    // iterate through ears, slicing them one by one
    let mut stop_idx = ear_idx;
    let mut prev_idx = 0;
    let mut next_idx = node!(ll, ear_idx).next_idx;
    while stop_idx != next_idx {
        prev_idx = node!(ll, ear_idx).prev_idx;
        next_idx = node!(ll, ear_idx).next_idx;
        if is_ear(ll, prev_idx, ear_idx, next_idx) {
            triangles.push(node!(ll, prev_idx).i);
            triangles.push(node!(ll, ear_idx).i);
            triangles.push(node!(ll, next_idx).i);
            ll.remove_node(ear_idx);
            // skipping the next vertex leads to less sliver triangles
            ear_idx = node!(ll, next_idx).next_idx;
            stop_idx = ear_idx;
        } else {
            ear_idx = next_idx;
        }
    }

    if prev_idx == next_idx {
        return;
    };
    // if we looped through the whole remaining polygon and can't
    // find any more ears
    if pass == 0 {
        let tmp = filter_points(ll, next_idx, NULL);
        earcut_linked_unhashed(ll, tmp, triangles, 1);
    } else if pass == 1 {
        ear_idx = cure_local_intersections(ll, next_idx, triangles);
        earcut_linked_unhashed(ll, ear_idx, triangles, 2);
    } else if pass == 2 {
        split_earcut(ll, next_idx, triangles);
    }
}

// interlink polygon nodes in z-order
fn index_curve<T: Float + Display>(ll: &mut LinkedLists<T>, start: NodeIdx) {
    let invsize = ll.invsize;
    let mut p = start;
    loop {
        if node!(ll, p).z == 0 {
            nodemut!(ll, p).z = zorder(node!(ll, p).x, node!(ll, p).y, invsize);
        }
        nodemut!(ll, p).prevz_idx = node!(ll, p).prev_idx;
        nodemut!(ll, p).nextz_idx = node!(ll, p).next_idx;
        p = node!(ll, p).next_idx;
        if p == start {
            break;
        }
    }

    let pzi = prevz!(ll, start).idx;
    nodemut!(ll, pzi).nextz_idx = NULL;
    nodemut!(ll, start).prevz_idx = NULL;
    sort_linked(ll, start);
}

// Simon Tatham's linked list merge sort algorithm
// http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
fn sort_linked<T: Float + Display>(ll: &mut LinkedLists<T>, mut list: NodeIdx) {
    let mut p;
    let mut q;
    let mut e;
    let mut nummerges;
    let mut psize;
    let mut qsize;
    let mut insize = 1;
    let mut tail;

    loop {
        p = list;
        list = NULL;
        tail = NULL;
        nummerges = 0;

        while p != NULL {
            nummerges += 1;
            q = p;
            psize = 0;
            while q != NULL && psize < insize {
                psize += 1;
                q = noderef!(ll, q).nextz_idx;
            }
            qsize = insize;

            while psize > 0 || (qsize > 0 && q != NULL) {
                if psize > 0 && (qsize == 0 || q == NULL || noderef!(ll, p).z <= noderef!(ll, q).z)
                {
                    e = p;
                    p = noderef!(ll, p).nextz_idx;
                    psize -= 1;
                } else {
                    e = q;
                    q = noderef!(ll, q).nextz_idx;
                    qsize -= 1;
                }

                if tail != NULL {
                    nodemut!(ll, tail).nextz_idx = e;
                } else {
                    list = e;
                }

                nodemut!(ll, e).prevz_idx = tail;
                tail = e;
            }

            p = q;
        }

        nodemut!(ll, tail).nextz_idx = NULL;
        insize *= 2;
        if nummerges <= 1 {
            break;
        }
    }
}

// check whether a polygon node forms a valid ear with adjacent nodes
fn is_ear<T: Float + Display>(
    ll: &LinkedLists<T>,
    prev: NodeIdx,
    ear: NodeIdx,
    next: NodeIdx,
) -> bool {
    let (a, b, c) = (noderef!(ll, prev), noderef!(ll, ear), noderef!(ll, next));
    let zero = T::zero();
    match area(a, b, c) >= zero {
        true => false, // reflex, cant be ear
        false => !ll.iter(c.next_idx..a.idx).any(|p| {
            point_in_triangle(&a, &b, &c, &p)
                && (area(prevref!(ll, p.idx), &p, nextref!(ll, p.idx)) >= zero)
        }),
    }
}

// helper for is_ear_hashed. needs manual inline (rust 2018)
#[inline(always)]
fn earcheck<T: Float + Display>(
    a: &Node<T>,
    b: &Node<T>,
    c: &Node<T>,
    prev: &Node<T>,
    p: &Node<T>,
    next: &Node<T>,
) -> bool {
    let zero = T::zero();

    (p.idx != a.idx)
        && (p.idx != c.idx)
        && point_in_triangle(&a, &b, &c, &p)
        && area(&prev, &p, &next) >= zero
}

#[inline(always)]
fn is_ear_hashed<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    prev_idx: NodeIdx,
    ear_idx: NodeIdx,
    next_idx: NodeIdx,
) -> bool {
    let (prev, ear, next) = (
        &node!(ll, prev_idx).clone(),
        &node!(ll, ear_idx).clone(),
        &node!(ll, next_idx).clone(),
    );
    let zero = T::zero();

    if area(prev, ear, next) >= zero {
        return false;
    };

    let bbox_maxx = T::max(prev.x, T::max(ear.x, next.x));
    let bbox_maxy = T::max(prev.y, T::max(ear.y, next.y));
    let bbox_minx = T::min(prev.x, T::min(ear.x, next.x));
    let bbox_miny = T::min(prev.y, T::min(ear.y, next.y));
    // z-order range for the current triangle bbox;
    let min_z = zorder(bbox_minx, bbox_miny, ll.invsize);
    let max_z = zorder(bbox_maxx, bbox_maxy, ll.invsize);

    let mut p = ear.prevz_idx;
    let mut n = ear.nextz_idx;
    while (p != NULL) && (node!(ll, p).z >= min_z) && (n != NULL) && (node!(ll, n).z <= max_z) {
        if earcheck(
            prev,
            ear,
            next,
            prevref!(ll, p),
            noderef!(ll, p),
            nextref!(ll, p),
        ) {
            return false;
        }
        p = node!(ll, p).prevz_idx;

        if earcheck(
            prev,
            ear,
            next,
            prevref!(ll, n),
            noderef!(ll, n),
            nextref!(ll, n),
        ) {
            return false;
        }
        n = node!(ll, n).nextz_idx;
    }

    nodemut!(ll, NULL).z = min_z - 1;
    while node!(ll, p).z >= min_z {
        if earcheck(
            prev,
            ear,
            next,
            prevref!(ll, p),
            noderef!(ll, p),
            nextref!(ll, p),
        ) {
            return false;
        }
        p = node!(ll, p).prevz_idx;
    }

    nodemut!(ll, NULL).z = max_z + 1;
    while node!(ll, n).z <= max_z {
        if earcheck(
            prev,
            ear,
            next,
            prevref!(ll, n),
            noderef!(ll, n),
            nextref!(ll, n),
        ) {
            return false;
        }
        n = node!(ll, n).nextz_idx;
    }

    true
}

fn filter_points<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    start: NodeIdx,
    mut end: NodeIdx,
) -> NodeIdx {
    dlog!(
        4,
        "fn filter_points, eliminate colinear or duplicate points"
    );
    if end == NULL {
        end = start;
    }
    if end >= ll.nodes.len() || start >= ll.nodes.len() {
        return NULL;
    }

    let mut p = start;
    let mut again;
    let zero = T::zero();

    // this loop "wastes" calculations by going over the same points multiple
    // times. however, altering the location of the 'end' node can disrupt
    // the algorithm of other code that calls the filter_points function.
    loop {
        again = false;
        if !node!(ll, p).steiner
            && (equals(noderef!(ll, p), nextref!(ll, p))
                || area(prevref!(ll, p), noderef!(ll, p), nextref!(ll, p)) == zero)
        {
            ll.remove_node(p);
            end = node!(ll, p).prev_idx;
            p = end;
            if p == node!(ll, p).next_idx {
                break end;
            }
            again = true;
        } else {
            p = node!(ll, p).next_idx;
        }
        if !again && p == end {
            break end;
        }
    }
}

// create a circular doubly linked list from polygon points in the
// specified winding order
fn linked_list<T: Float + Display>(
    data: &Vec<T>,
    start: usize,
    end: usize,
    clockwise: bool,
) -> (LinkedLists<T>, NodeIdx) {
    let mut ll: LinkedLists<T> = LinkedLists::new(data.len() / DIM);
    if data.len() < 80 {
        ll.usehash = false
    };
    let (last_idx, _) = linked_list_add_contour(&mut ll, data, start, end, clockwise);
    (ll, last_idx)
}

// add new nodes to an existing linked list.
fn linked_list_add_contour<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    data: &Vec<T>,
    start: usize,
    end: usize,
    clockwise: bool,
) -> (NodeIdx, NodeIdx) {
    if start > data.len() || end > data.len() || data.len() == 0 {
        return (NULL, NULL);
    }
    let mut lastidx = NULL;
    let mut leftmost_idx = NULL;
    let mut contour_minx = T::max_value();

    if clockwise == (signed_area(&data, start, end) > T::zero()) {
        for i in (start..end).step_by(DIM) {
            lastidx = ll.insert_node(i / DIM, data[i], data[i + 1], lastidx);
            if contour_minx > data[i] {
                contour_minx = data[i];
                leftmost_idx = lastidx
            };
            if ll.usehash {
                ll.miny = T::min(data[i + 1], ll.miny);
                ll.maxx = T::max(data[i], ll.maxx);
                ll.maxy = T::max(data[i + 1], ll.maxy);
            }
        }
    } else {
        for i in (start..=(end - DIM)).rev().step_by(DIM) {
            lastidx = ll.insert_node(i / DIM, data[i], data[i + 1], lastidx);
            if contour_minx > data[i] {
                contour_minx = data[i];
                leftmost_idx = lastidx
            };
            if ll.usehash {
                ll.miny = T::min(data[i + 1], ll.miny);
                ll.maxx = T::max(data[i], ll.maxx);
                ll.maxy = T::max(data[i + 1], ll.maxy);
            }
        }
    }

    ll.minx = T::min(contour_minx, ll.minx);

    if equals(noderef!(ll, lastidx), nextref!(ll, lastidx)) {
        ll.remove_node(lastidx);
        lastidx = noderef!(ll, lastidx).next_idx;
    }
    return (lastidx, leftmost_idx);
}

// z-order of a point given coords and inverse of the longer side of
// data bbox
#[inline(always)]
fn zorder<T: Float + Display>(xf: T, yf: T, invsize: T) -> i32 {
    // coords are transformed into non-negative 15-bit integer range
    // stored in two 32bit ints, which are combined into a single 64 bit int.
    let x: i64 = num_traits::cast::<T, i64>(xf * invsize).unwrap();
    let y: i64 = num_traits::cast::<T, i64>(yf * invsize).unwrap();
    let mut xy: i64 = x << 32 | y;

    // todo ... big endian?
    xy = (xy | (xy << 8)) & 0x00FF00FF00FF00FF;
    xy = (xy | (xy << 4)) & 0x0F0F0F0F0F0F0F0F;
    xy = (xy | (xy << 2)) & 0x3333333333333333;
    xy = (xy | (xy << 1)) & 0x5555555555555555;

    ((xy >> 32) | (xy << 1)) as i32
}

// check if a point lies within a convex triangle
fn point_in_triangle<T: Float + Display>(
    a: &Node<T>,
    b: &Node<T>,
    c: &Node<T>,
    p: &Node<T>,
) -> bool {
    let zero = T::zero();

    ((c.x - p.x) * (a.y - p.y) - (a.x - p.x) * (c.y - p.y) >= zero)
        && ((a.x - p.x) * (b.y - p.y) - (b.x - p.x) * (a.y - p.y) >= zero)
        && ((b.x - p.x) * (c.y - p.y) - (c.x - p.x) * (b.y - p.y) >= zero)
}

pub fn earcut<T: Float + Display>(
    data: &Vec<T>,
    hole_indices: &Vec<usize>,
    dims: usize,
) -> Vec<usize> {
    let outer_len = match hole_indices.len() {
        0 => data.len(),
        _ => hole_indices[0] * DIM,
    };

    let (mut ll, mut outer_node) = linked_list(data, 0, outer_len, true);
    let mut triangles: Vec<usize> = Vec::with_capacity(data.len() / DIM);
    if ll.nodes.len() == 1 || DIM != dims {
        return triangles;
    }

    outer_node = eliminate_holes(&mut ll, data, hole_indices, outer_node);

    if ll.usehash {
        ll.invsize = calc_invsize(ll.minx, ll.miny, ll.maxx, ll.maxy);

        // translate all points so min is 0,0. prevents subtraction inside
        // zorder. also note invsize does not depend on translation in space
        // if one were translating in a space with an even spaced grid of points.
        // floating point space is not evenly spaced, but it is close enough for
        // this hash algorithm
        let (mx, my) = (ll.minx, ll.miny);
        ll.nodes.iter_mut().for_each(|n| n.x = n.x - mx);
        ll.nodes.iter_mut().for_each(|n| n.y = n.y - my);
        earcut_linked_hashed(&mut ll, outer_node, &mut triangles, 0);
    } else {
        earcut_linked_unhashed(&mut ll, outer_node, &mut triangles, 0);
    }

    triangles
}

// signed area of a parallelogram
fn area<T: Float + Display>(p: &Node<T>, q: &Node<T>, r: &Node<T>) -> T {
    (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
}

// check if two points are equal
fn equals<T: Float + Display>(p1: &Node<T>, p2: &Node<T>) -> bool {
    p1.x == p2.x && p1.y == p2.y
}

/* go through all polygon nodes and cure small local self-intersections
what is a small local self-intersection? well, lets say you have four points
a,b,c,d. now imagine you have three line segments, a-b, b-c, and c-d. now
imagine two of those segments overlap each other. thats an intersection. so
this will remove one of those nodes so there is no more overlap.

but theres another important aspect of this function. it will dump triangles
into the 'triangles' variable, thus this is part of the triangulation
algorithm itself.*/
fn cure_local_intersections<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    instart: NodeIdx,
    triangles: &mut Vec<NodeIdx>,
) -> NodeIdx {
    let mut p = instart;
    let mut start = instart;

    //        2--3  4--5 << 2-3 + 4-5 pseudointersects
    //           x  x
    //  0  1  2  3  4  5  6  7
    //  a  p  pn b
    //              eq     a      b
    //              psi    a p pn b
    //              li  pa a p pn b bn
    //              tp     a p    b
    //              rn       p pn
    //              nst    a      p pn b
    //                            st

    //
    //                            a p  pn b

    loop {
        let a = node!(ll, p).prev_idx;
        let b = next!(ll, p).next_idx;

        if !equals(noderef!(ll, a), noderef!(ll, b))
            && pseudo_intersects(
                noderef!(ll, a),
                noderef!(ll, p),
                nextref!(ll, p),
                noderef!(ll, b),
            )
			// prev next a, prev next b
            && locally_inside(ll, noderef!(ll, a), noderef!(ll, b))
            && locally_inside(ll, noderef!(ll, b), noderef!(ll, a))
        {
            triangles.push(noderef!(ll, a).i);
            triangles.push(noderef!(ll, p).i);
            triangles.push(noderef!(ll, b).i);

            // remove two nodes involved
            ll.remove_node(p);
            let nidx = noderef!(ll, p).next_idx;
            ll.remove_node(nidx);

            start = noderef!(ll, b).idx;
            p = start;
        }
        p = noderef!(ll, p).next_idx;
        if p == start {
            break;
        }
    }

    return p;
}

// try splitting polygon into two and triangulate them independently
fn split_earcut<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    start_idx: NodeIdx,
    triangles: &mut Vec<NodeIdx>,
) {
    // look for a valid diagonal that divides the polygon into two
    let mut a = start_idx;
    loop {
        let mut b = next!(ll, a).next_idx;
        while b != noderef!(ll, a).prev_idx {
            if noderef!(ll, a).i != noderef!(ll, b).i
                && is_valid_diagonal(ll, noderef!(ll, a), noderef!(ll, b))
            {
                // split the polygon in two by the diagonal
                let mut c = split_bridge_polygon(ll, a, b);

                // filter colinear points around the cuts
                let an = noderef!(ll, a).next_idx;
                let cn = noderef!(ll, c).next_idx;
                a = filter_points(ll, a, an);
                c = filter_points(ll, c, cn);

                // run earcut on each half
                earcut_linked_hashed(ll, a, triangles, 0);
                earcut_linked_hashed(ll, c, triangles, 0);
                return;
            }
            b = noderef!(ll, b).next_idx;
        }
        a = noderef!(ll, a).next_idx;
        if a == start_idx {
            break;
        }
    }
}

// find a bridge between vertices that connects hole with an outer ring
// and and link it
fn eliminate_hole<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    hole_idx: NodeIdx,
    outer_node_idx: NodeIdx,
) {
    let test_idx = find_hole_bridge(ll, hole_idx, outer_node_idx);
    let b = split_bridge_polygon(ll, test_idx, hole_idx);
    let ni = node!(ll, b).next_idx;
    filter_points(ll, b, ni);
}

// David Eberly's algorithm for finding a bridge between hole and outer polygon
fn find_hole_bridge<T: Float + Display>(
    ll: &LinkedLists<T>,
    hole: NodeIdx,
    outer_node: NodeIdx,
) -> NodeIdx {
    let mut p = outer_node;
    let hx = node!(ll, hole).x;
    let hy = node!(ll, hole).y;
    let mut qx = T::neg_infinity();
    let mut m: NodeIdx = NULL;

    // find a segment intersected by a ray from the hole's leftmost
    // point to the left; segment's endpoint with lesser x will be
    // potential connection point
    let calcx =
        |p: &Node<T>| p.x + (hy - p.y) * (next!(ll, p.idx).x - p.x) / (next!(ll, p.idx).y - p.y);
    for (p, n) in ll
        .iter_pairs(p..outer_node)
        .filter(|(p, n)| hy <= p.y && hy >= n.y)
        .filter(|(p, n)| n.y != p.y)
        .filter(|(p, _)| calcx(p) <= hx)
    {
        if qx < calcx(p) {
            qx = calcx(p);
            if qx == hx && hy == p.y {
                return p.idx;
            } else if qx == hx && hy == n.y {
                return p.next_idx;
            }
            m = if p.x < n.x { p.idx } else { n.idx };
        }
    }

    if m == NULL {
        return NULL;
    }

    // hole touches outer segment; pick lower endpoint
    if hx == qx {
        return prev!(ll, m).idx;
    }

    // look for points inside the triangle of hole point, segment
    // intersection and endpoint; if there are no points found, we have
    // a valid connection; otherwise choose the point of the minimum
    // angle with the ray as connection point

    let mp = Node::new(0, node!(ll, m).x, node!(ll, m).y, 0);
    p = next!(ll, m).idx;
    let x1 = if hy < mp.y { hx } else { qx };
    let x2 = if hy < mp.y { qx } else { hx };
    let n1 = Node::new(0, x1, hy, 0);
    let n2 = Node::new(0, x2, hy, 0);
    let two = num_traits::cast::<f64, T>(2.).unwrap();

    let calctan = |p: &Node<T>| (hy - p.y).abs() / (hx - p.x); // tangential
    ll.iter(p..m)
        .filter(|p| hx > p.x && p.x >= mp.x)
        .filter(|p| point_in_triangle(&n1, &mp, &n2, &p))
        .fold((m, T::max_value() / two), |(m, tan_min), p| {
            if ((calctan(p) < tan_min) || (calctan(p) == tan_min && p.x > noderef!(ll, m).x))
                && locally_inside(ll, &p, noderef!(ll, hole))
            {
                (p.idx, calctan(p))
            } else {
                (m, tan_min)
            }
        })
        .0
}

// check if a diagonal between two polygon nodes is valid (lies in
// polygon interior)
fn is_valid_diagonal<T: Float + Display>(ll: &LinkedLists<T>, a: &Node<T>, b: &Node<T>) -> bool {
    return next!(ll, a.idx).i != b.i
        && prev!(ll, a.idx).i != b.i
        && !intersects_polygon(ll, a, b)
        && locally_inside(ll, a, b)
        && locally_inside(ll, b, a)
        && middle_inside(ll, a, b);
}

/* check if two segments cross over each other. note this is different
from pure intersction. only two segments crossing over at some interior
point is considered intersection.

line segment p1-q1 vs line segment p2-q2.

note that if they are collinear, or if the end points touch, or if
one touches the other at one point, it is not considered an intersection.

please note that the other algorithms in this earcut code depend on this
interpretation of the concept of intersection - if this is modified
so that endpoint touching qualifies as intersection, then it will have
a problem with certain inputs.

bsed on https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

this has been modified from the version in earcut.js to remove the
detection for endpoint detection.

    a1=area(p1,q1,p2);a2=area(p1,q1,q2);a3=area(p2,q2,p1);a4=area(p2,q2,q1);
    p1 q1    a1 cw   a2 cw   a3 ccw   a4  ccw  a1==a2  a3==a4  fl
    p2 q2
    p1 p2    a1 ccw  a2 ccw  a3 cw    a4  cw   a1==a2  a3==a4  fl
    q1 q2
    p1 q2    a1 ccw  a2 ccw  a3 ccw   a4  ccw  a1==a2  a3==a4  fl
    q1 p2
    p1 q2    a1 cw   a2 ccw  a3 ccw   a4  cw   a1!=a2  a3!=a4  tr
    p2 q1
*/

fn pseudo_intersects<T: Float + Display>(
    p1: &Node<T>,
    q1: &Node<T>,
    p2: &Node<T>,
    q2: &Node<T>,
) -> bool {
    if (equals(p1, p2) && equals(q1, q2)) || (equals(p1, q2) && equals(q1, p2)) {
        return true;
    }
    let zero = T::zero();

    return (area(p1, q1, p2) > zero) != (area(p1, q1, q2) > zero)
        && (area(p2, q2, p1) > zero) != (area(p2, q2, q1) > zero);
}

// check if a polygon diagonal intersects any polygon segments
fn intersects_polygon<T: Float + Display>(ll: &LinkedLists<T>, a: &Node<T>, b: &Node<T>) -> bool {
    ll.iter_pairs(a.idx..a.idx).any(|(p, n)| {
        p.i != a.i && n.i != a.i && p.i != b.i && n.i != b.i && pseudo_intersects(&p, &n, a, b)
    })
}

// check if a polygon diagonal is locally inside the polygon
fn locally_inside<T: Float + Display>(ll: &LinkedLists<T>, a: &Node<T>, b: &Node<T>) -> bool {
    let zero = T::zero();

    match area(prevref!(ll, a.idx), a, nextref!(ll, a.idx)) < zero {
        true => area(a, b, nextref!(ll, a.idx)) >= zero && area(a, prevref!(ll, a.idx), b) >= zero,
        false => area(a, b, prevref!(ll, a.idx)) < zero || area(a, nextref!(ll, a.idx), b) < zero,
    }
}

// check if the middle point of a polygon diagonal is inside the polygon
fn middle_inside<T: Float + Display>(ll: &LinkedLists<T>, a: &Node<T>, b: &Node<T>) -> bool {
    let two = num_traits::cast::<f64, T>(2.0).unwrap();

    let (mx, my) = ((a.x + b.x) / two, (a.y + b.y) / two);
    ll.iter_pairs(a.idx..a.idx)
        .filter(|(p, n)| (p.y > my) != (n.y > my))
        .filter(|(p, n)| n.y != p.y)
        .filter(|(p, n)| (mx) < ((n.x - p.x) * (my - p.y) / (n.y - p.y) + p.x))
        .fold(false, |inside, _| !inside)
}

/* link two polygon vertices with a bridge;

if the vertices belong to the same linked list, this splits the list
into two new lists, representing two new polygons.

if the vertices belong to separate linked lists, it merges them into a
single linked list.

For example imagine 6 points, labeled with numbers 0 thru 5, in a single cycle.
Now split at points 1 and 4. The 2 new polygon cycles will be like this:
0 1 4 5 0 1 ...  and  1 2 3 4 1 2 3 .... However because we are using linked
lists of nodes, there will be two new nodes, copies of points 1 and 4. So:
the new cycles will be through nodes 0 1 4 5 0 1 ... and 2 3 6 7 2 3 6 7 .

splitting algorithm:

.0...1...2...3...4...5...     6     7
5p1 0a2 1m3 2n4 3b5 4q0      .c.   .d.

an<-2     an = a.next,
bp<-3     bp = b.prev;
1.n<-4    a.next = b;
4.p<-1    b.prev = a;
6.n<-2    c.next = an;
2.p<-6    an.prev = c;
7.n<-6    d.next = c;
6.p<-7    c.prev = d;
3.n<-7    bp.next = d;
7.p<-3    d.prev = bp;

result of split:
<0...1> <2...3> <4...5>      <6....7>
5p1 0a4 6m3 2n7 1b5 4q0      7c2  3d6
      x x     x x            x x  x x    // x shows links changed

a b q p a b q p  // begin at a, go next (new cycle 1)
a p q b a p q b  // begin at a, go prev (new cycle 1)
m n d c m n d c  // begin at m, go next (new cycle 2)
m c d n m c d n  // begin at m, go prev (new cycle 2)

Now imagine that we have two cycles, and
they are 0 1 2, and 3 4 5. Split at points 1 and
4 will result in a single, long cycle,
0 1 4 5 3 7 6 2 0 1 4 5 ..., where 6 and 1 have the
same x y f64s, as do 7 and 4.

 0...1...2   3...4...5        6     7
2p1 0a2 1m0 5n4 3b5 4q3      .c.   .d.

an<-2     an = a.next,
bp<-3     bp = b.prev;
1.n<-4    a.next = b;
4.p<-1    b.prev = a;
6.n<-2    c.next = an;
2.p<-6    an.prev = c;
7.n<-6    d.next = c;
6.p<-7    c.prev = d;
3.n<-7    bp.next = d;
7.p<-3    d.prev = bp;

result of split:
 0...1...2   3...4...5        6.....7
2p1 0a4 6m0 5n7 1b5 4q3      7c2   3d6
      x x     x x            x x   x x

a b q n d c m p a b q n d c m .. // begin at a, go next
a p m c d n q b a p m c d n q .. // begin at a, go prev

Return value.

Return value is the new node, at point 7.
*/
fn split_bridge_polygon<T: Float + Display>(
    ll: &mut LinkedLists<T>,
    a: NodeIdx,
    b: NodeIdx,
) -> NodeIdx {
    let cidx = ll.nodes.len();
    let didx = cidx + 1;
    let mut c = Node::new(
        noderef!(ll, a).i,
        noderef!(ll, a).x,
        noderef!(ll, a).y,
        cidx,
    );
    let mut d = Node::new(
        noderef!(ll, b).i,
        noderef!(ll, b).x,
        noderef!(ll, b).y,
        didx,
    );

    let an = noderef!(ll, a).next_idx;
    let bp = noderef!(ll, b).prev_idx;

    nodemut!(ll, a).next_idx = b;
    nodemut!(ll, b).prev_idx = a;

    c.next_idx = an;
    nodemut!(ll, an).prev_idx = cidx;

    d.next_idx = cidx;
    c.prev_idx = didx;

    nodemut!(ll, bp).next_idx = didx;
    d.prev_idx = bp;

    ll.nodes.push(c);
    ll.nodes.push(d);
    return didx;
}

// return a percentage difference between the polygon area and its
// triangulation area; used to verify correctness of triangulation
pub fn deviation<T: Float + Display>(
    data: &Vec<T>,
    hole_indices: &Vec<usize>,
    dims: usize,
    triangles: &Vec<usize>,
) -> T {
    if DIM != dims {
        return T::nan();
    }
    let mut indices = hole_indices.clone();
    indices.push(data.len() / DIM);
    let (ix, iy) = (indices.iter(), indices.iter().skip(1));
    let body_area = signed_area(&data, 0, indices[0] * DIM).abs();
    let polygon_area = ix.zip(iy).fold(body_area, |a, (ix, iy)| {
        a - signed_area(&data, ix * DIM, iy * DIM).abs()
    });

    let i = triangles.iter().skip(0).step_by(3).map(|x| x * DIM);
    let j = triangles.iter().skip(1).step_by(3).map(|x| x * DIM);
    let k = triangles.iter().skip(2).step_by(3).map(|x| x * DIM);
    let triangles_area = i.zip(j).zip(k).fold(T::zero(), |ta, ((a, b), c)| {
        ta + ((data[a] - data[c]) * (data[b + 1] - data[a + 1])
            - (data[a] - data[b]) * (data[c + 1] - data[a + 1]))
            .abs()
    });
    let zero = T::zero();

    match polygon_area == zero && triangles_area == zero {
        true => zero,
        false => ((triangles_area - polygon_area) / polygon_area).abs(),
    }
}

fn signed_area<T: Float + Display>(data: &Vec<T>, start: usize, end: usize) -> T {
    let i = (start..end).step_by(DIM);
    let j = (start..end).cycle().skip((end - DIM) - start).step_by(DIM);
    let zero = T::zero();
    i.zip(j).fold(zero, |s, (i, j)| {
        s + (data[j] - data[i]) * (data[i + 1] + data[j + 1])
    })
}

// turn a polygon in a multi-dimensional array form (e.g. as in GeoJSON)
// into a form Earcut accepts
pub fn flatten<T: Float + Display>(data: &Vec<Vec<Vec<T>>>) -> (Vec<T>, Vec<usize>, usize) {
    (
        data.iter().cloned().flatten().flatten().collect::<Vec<T>>(), // flat data
        data.iter()
            .take(data.len() - 1)
            .scan(0, |holeidx, v| {
                *holeidx += v.len();
                Some(*holeidx)
            })
            .collect::<Vec<usize>>(), // hole indexes
        data[0][0].len(),                                             // dimensions
    )
}

fn pn(a: usize) -> String {
    match a {
        0x777A91CC => String::from("NULL"),
        _ => a.to_string(),
    }
}
fn pb(a: bool) -> String {
    match a {
        true => String::from("x"),
        false => String::from(" "),
    }
}
fn dump<T: Float + Display>(ll: &LinkedLists<T>) -> String {
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

fn cycle_dump<T: Float + Display>(ll: &LinkedLists<T>, p: NodeIdx) -> String {
    let mut s = format!("cycle from {}, ", p);
    s.push_str(&format!(" len {}, idxs:", 0)); //cycle_len(&ll, p)));
    let mut i = p;
    let end = i;
    let mut count = 0;
    loop {
        count += 1;
        s.push_str(&format!("{} ", noderef!(ll, i).idx));
        s.push_str(&format!("(i:{}), ", noderef!(ll, i).i));
        i = noderef!(ll, i).next_idx;
        if i == end {
            break s;
        }
        if count > ll.nodes.len() {
            s.push_str(&format!(" infinite loop"));
            break s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycles_report<T: num_traits::float::Float + std::fmt::Display>(
        ll: &LinkedLists<T>,
    ) -> String {
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
                let end = noderef!(ll, p).prev_idx;
                markv[p] = cycler;
                let mut count = 0;
                loop {
                    p = noderef!(ll, p).next_idx;
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

    fn dump_cycle<T: num_traits::float::Float + std::fmt::Display>(
        ll: &LinkedLists<T>,
        start: usize,
    ) -> String {
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
            let n = noderef!(ll, idx).clone();
            state = 0; //horsh( state, n.i  as i32);
            s.push_str(&format!(
                " {:>3} {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2}\n",
                count,
                n.idx,
                n.i,
                prev!(ll, n.idx).i,
                next!(ll, n.idx).i,
                n.x,
                n.y,
                pn(n.prevz_idx),
                pn(n.nextz_idx),
                pb(n.steiner),
                //                pb(ll.freelist.contains(&n.idx)),
                false,
                cycle_len(&ll, n.idx),
            ));
            idx = next!(ll, idx).idx;
            count += 1;
            if idx == endidx || count > ll.nodes.len() {
                break;
            }
        }
        s.push_str(&format!("dump end, horshcount:{} horsh:{}", count, state));
        return s;
    }

    fn cycle_len<T: num_traits::float::Float + std::fmt::Display>(
        ll: &LinkedLists<T>,
        p: NodeIdx,
    ) -> usize {
        if p >= ll.nodes.len() {
            return 0;
        }
        let end = noderef!(ll, p).prev_idx;
        let mut i = p;
        let mut count = 1;
        loop {
            i = noderef!(ll, i).next_idx;
            count += 1;
            if i == end {
                break count;
            }
            if count > ll.nodes.len() {
                break count;
            }
        }
    }

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
    fn horsh_ll<T: num_traits::float::Float + std::fmt::Display>(
        ll: &LinkedLists<T>,
        starti: VertIdx,
    ) -> String {
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
            let n = noderef!(ll, idx).clone();
            state = horsh(state, n.i as u32);
            idx = next!(ll, idx).idx;
            count += 1;
            if idx == endidx || count > ll.nodes.len() {
                break;
            }
        }
        s.push_str(&format!(" count:{} horsh: {}", count, state));
        return s;
    }

    #[test]
    fn test_linked_list() {
        let data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&data, 0, data.len(), true);
        assert!(ll.nodes.len() == 5);
        assert!(ll.nodes[1].idx == 1);
        assert!(ll.nodes[1].i == 6 / DIM);
        assert!(ll.nodes[1].i == 3);
        assert!(ll.nodes[1].x == 1.0);
        assert!(ll.nodes[1].y == 0.0);
        assert!(ll.nodes[1].next_idx == 2 && ll.nodes[1].prev_idx == 4);
        assert!(ll.nodes[4].next_idx == 1 && ll.nodes[4].prev_idx == 3);
        ll.remove_node(2);
    }

    #[test]
    fn test_iter_pairs() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&data, 0, data.len(), true);
        let mut v: Vec<Node<f32>> = Vec::new();
        //        ll.iter(1..2)
        //.zip(ll.iter(2..3))
        ll.iter_pairs(1..2).for_each(|(p, n)| {
            v.push(p.clone());
            v.push(n.clone());
        });
        println!("{:?}", v);
        //		assert!(false);
    }

    #[test]
    fn test_point_in_triangle() {
        let data = vec![0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.1];
        let (ll, _) = linked_list(&data, 0, data.len(), true);
        assert!(point_in_triangle(
            &ll.nodes[1],
            &ll.nodes[2],
            &ll.nodes[3],
            &ll.nodes[4]
        ));
    }

    #[test]
    fn test_signed_area() {
        let data1 = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let data2 = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let a1 = signed_area(&data1, 0, 4);
        let a2 = signed_area(&data2, 0, 4);
        assert!(a1 == -a2);
    }

    #[test]
    fn test_deviation() {
        let data1 = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let tris = vec![0, 1, 2, 2, 3, 0];
        let hi: Vec<usize> = Vec::new();
        assert!(deviation(&data1, &hi, DIM, &tris) == 0.0);
    }

    #[test]
    fn test_split_bridge_polygon() {
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let hole = vec![0.1, 0.1, 0.1, 0.2, 0.2, 0.2];
        body.extend(hole);
        let (mut ll, _) = linked_list(&body, 0, body.len(), true);
        assert!(cycle_len(&ll, 1) == body.len() / DIM);
        let (left, right) = (1, 5);
        let np = split_bridge_polygon(&mut ll, left, right);
        assert!(cycle_len(&ll, left) == 4);
        assert!(cycle_len(&ll, np) == 5);
        // contrary to name, this should join the two cycles back together.
        let np2 = split_bridge_polygon(&mut ll, left, np);
        assert!(cycle_len(&ll, np2) == 11);
        assert!(cycle_len(&ll, left) == 11);
    }

    #[test]
    fn test_equals() {
        let body = vec![0.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&body, 0, body.len(), true);
        assert!(equals(&ll.nodes[1], &ll.nodes[2]));

        let body = vec![2.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&body, 0, body.len(), true);
        assert!(!equals(&ll.nodes[1], &ll.nodes[2]));
    }

    #[test]
    fn test_area() {
        let body = vec![4.0, 0.0, 4.0, 3.0, 0.0, 0.0]; // counterclockwise
        let (ll, _) = linked_list(&body, 0, body.len(), true);
        assert!(area(&ll.nodes[1], &ll.nodes[2], &ll.nodes[3]) == -12.0);
        let body2 = vec![4.0, 0.0, 0.0, 0.0, 4.0, 3.0]; // clockwise
        let (ll2, _) = linked_list(&body2, 0, body2.len(), true);
        // creation apparently modifies all winding to ccw
        assert!(area(&ll2.nodes[1], &ll2.nodes[2], &ll2.nodes[3]) == -12.0);
    }

    #[test]
    fn test_is_ear() {
        let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(!is_ear(&ll, 1, 2, 3));
        assert!(!is_ear(&ll, 2, 3, 1));
        assert!(!is_ear(&ll, 3, 1, 2));

        let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.4];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(is_ear(&ll, 4, 1, 2) == false);
        assert!(is_ear(&ll, 1, 2, 3) == true);
        assert!(is_ear(&ll, 2, 3, 4) == false);
        assert!(is_ear(&ll, 3, 4, 1) == true);

        let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(is_ear(&ll, 3, 1, 2));

        let m = vec![0.0, 0.0, 4.0, 0.0, 4.0, 3.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(is_ear(&ll, 3, 1, 2));
    }

    #[test]
    fn test_filter_points() {
        let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let lllen = ll.nodes.len();
        println!("len {}", ll.nodes.len());
        println!("{}", dump(&ll));
        let r1 = filter_points(&mut ll, 1, lllen - 1);
        println!("{}", dump(&ll));
        println!("r1 {} cyclen {}", r1, cycle_len(&ll, r1));
        assert!(cycle_len(&ll, r1) == 4);

        let n = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&n, 0, n.len(), true);
        let lllen = ll.nodes.len();
        let r2 = filter_points(&mut ll, 1, lllen - 1);
        assert!(cycle_len(&ll, r2) == 4);

        let n2 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&n2, 0, n2.len(), true);
        let r32 = filter_points(&mut ll, 1, 99);
        assert!(cycle_len(&ll, r32) != 4);

        let o = vec![0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5];
        let (mut ll, _) = linked_list(&o, 0, o.len(), true);
        let lllen = ll.nodes.len();
        let r3 = filter_points(&mut ll, 1, lllen - 1);
        assert!(cycle_len(&ll, r3) == 3);

        let o = vec![0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&o, 0, o.len(), true);
        let lllen = ll.nodes.len();
        let r3 = filter_points(&mut ll, 1, lllen - 1);
        assert!(cycle_len(&ll, r3) == 5);
    }

    #[test]
    fn test_earcut_linked() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let (mut tris, pass) = (Vec::new(), 0);
        earcut_linked_hashed(&mut ll, 1, &mut tris, pass);
        assert!(tris.len() == 6);

        let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let (mut tris, pass) = (Vec::new(), 0);
        earcut_linked_unhashed(&mut ll, 1, &mut tris, pass);
        assert!(tris.len() == 9);

        let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let (mut tris, pass) = (Vec::new(), 0);
        earcut_linked_hashed(&mut ll, 1, &mut tris, pass);
        assert!(tris.len() == 9);
    }

    #[test]
    fn test_middle_inside() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(middle_inside(&ll, noderef!(ll, 1), noderef!(ll, 3)));
        assert!(middle_inside(&ll, noderef!(ll, 2), noderef!(ll, 4)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(!middle_inside(&ll, noderef!(ll, 1), noderef!(ll, 3)));
        assert!(middle_inside(&ll, noderef!(ll, 2), noderef!(ll, 4)));
    }

    #[test]
    fn test_locally_inside() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(locally_inside(&ll, noderef!(ll, 1), noderef!(ll, 1)));
        assert!(locally_inside(&ll, noderef!(ll, 1), noderef!(ll, 2)));
        assert!(locally_inside(&ll, noderef!(ll, 1), noderef!(ll, 3)));
        assert!(locally_inside(&ll, noderef!(ll, 1), noderef!(ll, 4)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(locally_inside(&ll, noderef!(ll, 1), noderef!(ll, 1)));
        assert!(locally_inside(&ll, noderef!(ll, 1), noderef!(ll, 2)));
        assert!(!locally_inside(&ll, noderef!(ll, 1), noderef!(ll, 3)));
        assert!(locally_inside(&ll, noderef!(ll, 1), noderef!(ll, 4)));
    }

    #[test]
    fn test_intersects_polygon() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);

        assert!(false == intersects_polygon(&ll, noderef!(ll, 0), noderef!(ll, 2)));
        assert!(false == intersects_polygon(&ll, noderef!(ll, 2), noderef!(ll, 0)));
        assert!(false == intersects_polygon(&ll, noderef!(ll, 1), noderef!(ll, 3)));
        assert!(false == intersects_polygon(&ll, noderef!(ll, 3), noderef!(ll, 1)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        dlog!(9, "{}", dump(&ll));
        dlog!(
            5,
            "{}",
            intersects_polygon(&ll, noderef!(ll, 0), noderef!(ll, 2))
        );
        dlog!(
            5,
            "{}",
            intersects_polygon(&ll, noderef!(ll, 2), noderef!(ll, 0))
        );
    }

    #[test]
    fn test_intersects_itself() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 0.9, 0.9, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        macro_rules! ti {
            ($ok:expr,$a:expr,$b:expr,$c:expr,$d:expr) => {
                assert!(
                    $ok == pseudo_intersects(
                        &ll.nodes[$a],
                        &ll.nodes[$b],
                        &ll.nodes[$c],
                        &ll.nodes[$d]
                    )
                );
            };
        };
        ti!(false, 0 + 1, 2 + 1, 0 + 1, 1 + 1);
        ti!(false, 0 + 1, 2 + 1, 1 + 1, 2 + 1);
        ti!(false, 0 + 1, 2 + 1, 2 + 1, 3 + 1);
        ti!(false, 0 + 1, 2 + 1, 3 + 1, 0 + 1);
        ti!(true, 0 + 1, 2 + 1, 3 + 1, 1 + 1);
        ti!(true, 0 + 1, 2 + 1, 1 + 1, 3 + 1);
        ti!(true, 2 + 1, 0 + 1, 3 + 1, 1 + 1);
        ti!(true, 2 + 1, 0 + 1, 1 + 1, 3 + 1);
        ti!(false, 0 + 1, 1 + 1, 2 + 1, 3 + 1);
        ti!(false, 1 + 1, 0 + 1, 2 + 1, 3 + 1);
        ti!(false, 0 + 1, 0 + 1, 2 + 1, 3 + 1);
        ti!(false, 0 + 1, 1 + 1, 3 + 1, 2 + 1);
        ti!(false, 1 + 1, 0 + 1, 3 + 1, 2 + 1);

        ti!(true, 0 + 1, 2 + 1, 2 + 1, 0 + 1); // special cases
        ti!(true, 0 + 1, 2 + 1, 0 + 1, 2 + 1);

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(false == pseudo_intersects(&ll.nodes[4], &ll.nodes[5], &ll.nodes[1], &ll.nodes[3]));

        // special case
        assert!(true == pseudo_intersects(&ll.nodes[4], &ll.nodes[5], &ll.nodes[3], &ll.nodes[1]));
    }

    #[test]
    fn test_is_valid_diagonal() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(!is_valid_diagonal(&ll, &ll.nodes[1], &ll.nodes[2]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[2], &ll.nodes[3]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[3], &ll.nodes[4]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[4], &ll.nodes[1]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[1], &ll.nodes[3]));
        assert!(is_valid_diagonal(&ll, &ll.nodes[2], &ll.nodes[4]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[3], &ll.nodes[4]));
        assert!(is_valid_diagonal(&ll, &ll.nodes[4], &ll.nodes[2]));
    }

    #[test]
    fn test_find_hole_bridge() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let hole_idx = ll.insert_node(0, 0.8, 0.8, NULL);
        assert!(1 == find_hole_bridge(&ll, hole_idx, 1));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.4, 0.5];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let hole_idx = ll.insert_node(0, 0.5, 0.5, NULL);
        assert!(5 == find_hole_bridge(&ll, hole_idx, 1));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -0.4, 0.5];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let hole_idx = ll.insert_node(0, 0.5, 0.5, NULL);
        assert!(5 == find_hole_bridge(&ll, hole_idx, 1));

        let m = vec![
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -0.1, 0.9, 0.1, 0.8, -0.1, 0.7, 0.1, 0.6, -0.1,
            0.5,
        ];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let hole_idx = ll.insert_node(0, 0.5, 0.9, NULL);
        assert!(5 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.1, NULL);
        assert!(9 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.5, NULL);
        assert!(9 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.55, NULL);
        assert!(9 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.6, NULL);
        assert!(8 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.65, NULL);
        assert!(7 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.7, NULL);
        assert!(7 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.75, NULL);
        assert!(7 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.8, NULL);
        assert!(6 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.85, NULL);
        assert!(5 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.9, NULL);
        assert!(5 == find_hole_bridge(&ll, hole_idx, 1));
        let hole_idx = ll.insert_node(0, 0.2, 0.95, NULL);
        assert!(5 == find_hole_bridge(&ll, hole_idx, 1));
    }

    #[test]
    fn test_eliminate_hole() {
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

        let hole = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        let (mut ll, _) = linked_list(&body, 0, bodyend, true);
        linked_list_add_contour(&mut ll, &body, holestart, holeend, false);
        assert!(cycle_len(&ll, 1) == 4);
        assert!(cycle_len(&ll, 5) == 4);
        eliminate_hole(&mut ll, holestart / DIM + 1, 1);
        println!("{}", dump(&ll));
        println!("{}", cycle_len(&ll, 1));
        println!("{}", cycle_len(&ll, 7));
        assert!(cycle_len(&ll, 1) == 10);

        let hole = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        linked_list_add_contour(&mut ll, &body, holestart, holeend, false);
        assert!(cycle_len(&ll, 1) == 10);
        assert!(cycle_len(&ll, 5) == 10);
        assert!(cycle_len(&ll, 11) == 4);
        eliminate_hole(&mut ll, 11, 2);
        assert!(!cycle_len(&ll, 1) != 10);
        assert!(!cycle_len(&ll, 1) != 10);
        assert!(!cycle_len(&ll, 5) != 10);
        assert!(!cycle_len(&ll, 10) != 4);
        assert!(cycle_len(&ll, 1) == 16);
        assert!(cycle_len(&ll, 1) == 16);
        assert!(cycle_len(&ll, 10) == 16);
        assert!(cycle_len(&ll, 15) == 16);
    }

    #[test]
    fn test_cycle_len() {
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.1, 0.1];

        let hole = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        let (mut ll, _) = linked_list(&body, 0, bodyend, true);
        linked_list_add_contour(&mut ll, &body, holestart, holeend, false);

        let hole = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        linked_list_add_contour(&mut ll, &body, holestart, holeend, false);

        dlog!(5, "{}", dump(&ll));
        dlog!(5, "{}", cycles_report(&ll));
    }

    #[test]
    fn test_cycles_report() {
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.1, 0.1];

        let hole = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        let (mut ll, _) = linked_list(&body, 0, bodyend, true);
        linked_list_add_contour(&mut ll, &body, holestart, holeend, false);

        let hole = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        linked_list_add_contour(&mut ll, &body, holestart, holeend, false);

        dlog!(5, "{}", dump(&ll));
        dlog!(5, "{}", cycles_report(&ll));
    }

    #[test]
    fn test_eliminate_holes() {
        let mut hole_indices: Vec<usize> = Vec::new();
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&body, 0, body.len(), true);
        let hole1 = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
        let hole2 = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8];
        hole_indices.push(body.len() / DIM);
        hole_indices.push((body.len() + hole1.len()) / DIM);
        body.extend(hole1);
        body.extend(hole2);

        eliminate_holes(&mut ll, &body, &hole_indices, 0);
    }

    #[test]
    fn test_cure_local_intersections() {
        // first test . it will not be able to detect the crossover
        // so it will not change anything.
        let m = vec![
            0.0, 0.0, 1.0, 0.0, 1.1, 0.1, 0.9, 0.1, 1.0, 0.05, 1.0, 1.0, 0.0, 1.0,
        ];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let mut triangles: Vec<usize> = Vec::new();
        cure_local_intersections(&mut ll, 0, &mut triangles);
        assert!(cycle_len(&ll, 1) == 7);
        assert!(triangles.len() == 0);

        // second test - we have three points that immediately cause
        // self intersection. so it should, in theory, detect and clean
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.1, 0.1, 1.1, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let mut triangles: Vec<usize> = Vec::new();
        cure_local_intersections(&mut ll, 1, &mut triangles);
        assert!(cycle_len(&ll, 1) == 4);
        assert!(triangles.len() == 3);
    }

    #[test]
    fn test_split_earcut() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let start = 1;
        let mut triangles: Vec<usize> = Vec::new();
        split_earcut(&mut ll, start, &mut triangles);
        assert!(triangles.len() == 6);
        assert!(ll.nodes.len() == 7);

        let m = vec![
            0.0, 0.0, 1.0, 0.0, 1.5, 0.5, 2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0, 1.5, 0.6, 1.0,
            1.0, 0.0, 1.0,
        ];
        let (mut ll, _) = linked_list(&m, 0, m.len(), true);
        let start = 1;
        let mut triangles: Vec<usize> = Vec::new();
        split_earcut(&mut ll, start, &mut triangles);
        assert!(ll.nodes.len() == 13);
    }

    #[test]
    fn test_flatten() {
        let data: Vec<Vec<Vec<f64>>> = vec![
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
            ],
            vec![
                vec![0.1, 0.1],
                vec![0.9, 0.1],
                vec![0.9, 0.9],
                vec![0.1, 0.9],
            ],
            vec![
                vec![0.2, 0.2],
                vec![0.8, 0.2],
                vec![0.8, 0.8],
                vec![0.2, 0.8],
            ],
        ];
        let (coords, hole_indices, dims) = flatten(&data);
        assert!(DIM == dims);
        println!("{:?} {:?}", coords, hole_indices);
        assert!(coords.len() == 24);
        assert!(hole_indices.len() == 2);
        assert!(hole_indices[0] == 4);
        assert!(hole_indices[1] == 8);
    }

    #[test]
    fn test_iss45() {
        let data = vec![
            vec![
                vec![10.0, 10.0],
                vec![25.0, 10.0],
                vec![25.0, 40.0],
                vec![10.0, 40.0],
            ],
            vec![vec![15.0, 30.0], vec![20.0, 35.0], vec![10.0, 40.0]],
            vec![vec![15.0, 15.0], vec![15.0, 20.0], vec![20.0, 15.0]],
        ];
        let (coords, hole_indices, dims) = flatten(&data);
        assert!(DIM == dims);
        let triangles = earcut(&coords, &hole_indices, DIM);
        assert!(triangles.len() > 4);
    }
}
