#![allow(dead_code)]

static NULL: usize = 0x777A91CC;
static NULL32: u32 = 0xFFFFFFFF;
//static DEBUG: usize = 4;
static DEBUG: usize = 0;
static TESTHASH: bool = true;

type NodeIdx = usize;

impl Node {
    fn new(i: usize, x: f64, y: f64, idx: usize) -> Node {
        Node {
            i: i,
            x: x,
            y: y,
            prev_idx: NULL,
            next_idx: NULL,
            z: NULL32,
            nextz_idx: NULL,
            prevz_idx: NULL,
            steiner: false,
            idx: idx,
        }
    }
}

#[derive(Clone)]
struct Node {
    i: usize,        // vertex index in f64s array
    x: f64,          // vertex x f64s
    y: f64,          // vertex y f64s
    prev_idx: usize, // previous vertex nodes in a polygon ring
    next_idx: usize,
    z: u32,           // z-order curve value
    prevz_idx: usize, // previous and next nodes in z-order
    nextz_idx: usize,
    steiner: bool, // indicates whether this is a steiner point
    idx: usize,    // index within vector that holds all nodes
}

macro_rules! dlog {
	($loglevel:expr, $($s:expr),*) => (
		if DEBUG>=$loglevel { print!("{}:",$loglevel); println!($($s),+); }
	)
}
macro_rules! node {
    ($ll:expr,$idx:expr) => {
        $ll.nodes[$idx]
    };
}
// Note: none of the following macros work for Left-Hand-Side of assignment.
macro_rules! next {
    ($ll:ident,$idx:expr) => {
        $ll.nodes[$ll.nodes[$idx].next_idx]
    };
}
macro_rules! prev {
    ($ll:ident,$idx:expr) => {
        $ll.nodes[$ll.nodes[$idx].prev_idx]
    };
}
macro_rules! prevz {
    ($ll:ident,$idx:expr) => {
        $ll.nodes[$ll.nodes[$idx].prevz_idx]
    };
}

pub struct LinkedLists {
    nodes: Vec<Node>,
    freelist: Vec<NodeIdx>, // remove nodes have their index stored here
}

#[derive(Debug)]
struct BoundingBox {
    minx: f64,
    miny: f64,
    maxx: f64,
    maxy: f64,
}
impl BoundingBox {
    fn expand(&mut self, n: &Node) {
        self.maxx = f64::max(self.maxx, n.x);
        self.maxy = f64::max(self.maxy, n.y);
        self.minx = f64::min(self.minx, n.x);
        self.miny = f64::min(self.miny, n.y);
    }
    fn new(p:&Node)->BoundingBox {
        BoundingBox {
            maxx: p.x,
            maxy: p.y,
            minx: p.x,
            miny: p.y,
        }
    }
}

impl LinkedLists {
    fn iter_range(&self, r: std::ops::Range<NodeIdx>) -> NodeIterator {
        return NodeIterator::new(self, r.start, r.end);
    }
    fn insert_node(&mut self, i: usize, x: f64, y: f64, last: NodeIdx) -> NodeIdx {
        dlog!(9, "insert_node {} {} {} {}", i, x, y, last);
        let mut p = Node::new(i, x, y, self.nodes.len());
        if last == NULL {
            p.next_idx = p.idx;
            p.prev_idx = p.idx;
        } else {
            p.next_idx = node!(self, last).next_idx;
            p.prev_idx = last;
            let lastnextidx = node!(self, last).next_idx;
            node!(self, lastnextidx).prev_idx = p.idx;
            node!(self, last).next_idx = p.idx;
        }
        self.nodes.push(p.clone());
        return p.idx;
    }
    fn remove_node(&mut self, p: NodeIdx) {
        if p == NULL {
            return;
        }

        let nx = node!(self, p).next_idx;
        let pr = node!(self, p).prev_idx;
        node!(self, nx).prev_idx = pr;
        node!(self, pr).next_idx = nx;

        let prz = node!(self, p).prevz_idx;
        let nxz = node!(self, p).nextz_idx;
        if prz != NULL {
            node!(self, prz).nextz_idx = nxz;
        }
        if nxz != NULL {
            node!(self, nxz).prevz_idx = prz;
        }
        self.freelist.push(p);
    }
    fn new() -> LinkedLists {
        LinkedLists {
            nodes: Vec::new(),
            freelist: Vec::new(),
        }
    }
} // LinkedLists

struct NodeIterator<'a> {
    start: NodeIdx,
    cur: NodeIdx,
    end: NodeIdx,
    count: usize,
    ll: &'a LinkedLists,
}

impl<'a> NodeIterator<'a> {
    fn new(ll: &LinkedLists, start: usize, end: usize) -> NodeIterator {
        NodeIterator {
            start: start,
            cur: start,
            end: end,
            count: 0,
            ll,
        }
    }
}

impl<'a> Iterator for NodeIterator<'a> {
    type Item = &'a Node;
    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.count == 0 {
            Some(&node!(self.ll, self.cur)) // single-node iterator
        } else if self.cur == std::usize::MAX {
            None // NULL node
        } else if self.cur == self.end {
            None // end of iteration
        } else {
            Some(&node!(self.ll, self.cur)) // normal iteration
        };
        self.count += 1;
        self.cur = node!(self.ll, self.cur).next_idx;
        result
    }
}

fn compare_x(a: &Node, b: &Node) -> std::cmp::Ordering {
    return a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal);
}

// link every hole into the outer loop, producing a single-ring polygon
// without holes
fn eliminate_holes(
    ll: &mut LinkedLists,
    data: &Vec<f64>,
    hole_indices: &Vec<usize>,
    inouter_node: NodeIdx,
    dim: usize,
) -> NodeIdx {
    let mut outer_node = inouter_node;
    let mut queue: Vec<Node> = Vec::new();
    let hlen = hole_indices.len();
    for i in 0..hlen {
        let start = hole_indices[i] * dim;
        let end = if i < (hlen - 1) {
            hole_indices[i + 1] * dim
        } else {
            data.len()
        };
        let list = linked_list_add_contour(ll, &data, start, end, dim, false);
        if list == node!(ll, list).next_idx {
            ll.nodes[list].steiner = true;
        }
        let leftmost = ll
            .iter_range(list..list)
            .min_by(|p, q| compare_x(p, q))
            .unwrap();

        queue.push(leftmost.clone());
    }

    queue.sort_by(compare_x);

    // process holes from left to right
    for i in 0..queue.len() {
        eliminate_hole(ll, queue[i].idx, outer_node);
        let nextidx = next!(ll, outer_node).idx;
        outer_node = filter_points(ll, outer_node, nextidx);
    }
    return outer_node;
} // elim holes

// minx, miny and invsize are later used to transform coords
// into integers for z-order calculation
fn calc_invsize(minx: f64, miny: f64, maxx: f64, maxy: f64) -> f64 {
    let invsize = f64::max(maxx - minx, maxy - miny);
    match invsize == 0.0 {
        true => 0.0,
        false => 1.0 / invsize,
    }
}

// main ear slicing loop which triangulates a polygon (given as a linked
// list)
fn earcut_linked(
    ll: &mut LinkedLists,
    mut ear: NodeIdx,
    triangles: &mut Vec<usize>,
    dim: usize,
    minx: f64,
    miny: f64,
    invsize: f64,
    pass: usize,
) {
    if ear == NULL {
        return;
    }

    // interlink polygon nodes in z-order
    // note this does nothing for smaller data len, b/c invsize will be 0
    if pass == 0 && (invsize > 0.0 || TESTHASH) {
        index_curve(ll, ear, minx, miny, invsize);
    }

    let mut stop = ear;
    let mut prev = 0;
    let mut next = 0;
    // iterate through ears, slicing them one by one
    while node!(ll, ear).prev_idx != node!(ll, ear).next_idx {
        dlog!(9, "p{} e{} n{} s{}", prev, ear, next, stop);
        prev = node!(ll, ear).prev_idx;
        next = node!(ll, ear).next_idx;

        let test;
        if invsize > 0.0 {
            test = is_ear_hashed(ll, ear, minx, miny, invsize);
        } else {
            test = is_ear(ll, ear);
        }
        if TESTHASH {
            assert!(is_ear(ll, ear) == is_ear_hashed(ll, ear, minx, miny, invsize));
        }
        if test {
            // cut off the triangle
            triangles.push(ll.nodes[prev].i / dim);
            triangles.push(ll.nodes[ear].i / dim);
            triangles.push(ll.nodes[next].i / dim);

            ll.remove_node(ear);

            // skipping the next vertex leads to less sliver triangles
            ear = ll.nodes[next].next_idx;
            stop = ll.nodes[next].next_idx;
            continue;
        }

        ear = next;

        // if we looped through the whole remaining polygon and can't
        // find any more ears
        if ear == stop {
            if pass == 0 {
                // try filtering points and slicing again
                let tmp = filter_points(ll, ear, NULL);
                earcut_linked(ll, tmp, triangles, dim, minx, miny, invsize, 1);
            } else if pass == 1 {
                // if this didn't work, try curing all small
                // self-intersections locally
                ear = cure_local_intersections(ll, ear, triangles, dim);
                earcut_linked(ll, ear, triangles, dim, minx, miny, invsize, 2);
            } else if pass == 2 {
                // as a last resort, try splitting the remaining polygon
                // into two
                split_earcut(ll, ear, triangles, dim, minx, miny, invsize);
            }
            break;
        }
    } // while
    dlog!(4, "earcut_linked end");
}

// interlink polygon nodes in z-order
fn index_curve(ll: &mut LinkedLists, start: NodeIdx, minx: f64, miny: f64, invsize: f64) {
    let mut p = start;
    loop {
        if node!(ll, p).z == NULL32 {
            node!(ll, p).z = zorder(node!(ll, p).x, node!(ll, p).y, minx, miny, invsize);
        }
        node!(ll, p).prevz_idx = node!(ll, p).prev_idx;
        node!(ll, p).nextz_idx = node!(ll, p).next_idx;
        p = node!(ll, p).next_idx;
        if p == start {
            break;
        }
    }

    let pzi = prevz!(ll, p).idx;
    node!(ll, pzi).nextz_idx = NULL;
    node!(ll, p).prevz_idx = NULL;

    sort_linked(ll, p);
}

// Simon Tatham's linked list merge sort algorithm
// http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
fn sort_linked(ll: &mut LinkedLists, inlist: NodeIdx) {
    let mut p;
    let mut q;
    let mut e;
    let mut nummerges;
    let mut psize;
    let mut qsize;
    let mut insize = 1;
    let mut list = inlist;
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
            for _ in 0..insize {
                psize += 1;
                q = node!(ll, q).nextz_idx;
                if q == NULL {
                    break;
                }
            }
            qsize = insize;

            while psize > 0 || (qsize > 0 && q != NULL) {
                if psize != 0 && (qsize == 0 || q == NULL || node!(ll, p).z <= node!(ll, q).z) {
                    e = p;
                    p = ll.nodes[p].nextz_idx;
                    psize -= 1;
                } else {
                    e = q;
                    q = ll.nodes[q].nextz_idx;
                    qsize -= 1;
                }

                if tail != NULL {
                    ll.nodes[tail].nextz_idx = e;
                } else {
                    list = e;
                }

                ll.nodes[e].prevz_idx = tail;
                tail = e;
            }

            p = q;
        }

        ll.nodes[tail].nextz_idx = NULL;
        insize *= 2;
        if nummerges <= 1 {
            break;
        }
    }
}

// check whether a polygon node forms a valid ear with adjacent nodes
fn is_ear(ll: &LinkedLists, ear: usize) -> bool {
    let (a, b, c) = (&prev!(ll, ear), &node!(ll, ear), &next!(ll, ear));
    match area(a, b, c) >= 0.0 {
        true => false, // reflex, cant be ear
        false => !ll.iter_range(c.next_idx..a.idx).any(|p| {
            point_in_triangle(&a, &b, &c, &p)
                && (area(&prev!(ll, p.idx), &p, &next!(ll, p.idx)) >= 0.0)
        }),
    }
}

fn is_ear_hashed(ll: &mut LinkedLists, ear: usize, minx: f64, miny: f64, invsize: f64) -> bool {
    let (a, b, c) = (&prev!(ll, ear), &node!(ll, ear), &next!(ll, ear));
    if area(&a, &b, &c) >= 0.0 {
        dlog!(9, "reflex, can't be an ear");
        return false;
    }

    let mut bbox = BoundingBox::new(&a);
	bbox.expand(&b);
	bbox.expand(&c);

    // z-order range for the current triangle bbox;
    let min_z = zorder(bbox.minx, bbox.miny, minx, miny, invsize);
    let max_z = zorder(bbox.maxx, bbox.maxy, minx, miny, invsize);

    let mut p = node!(ll, ear).prevz_idx;
    let mut n = node!(ll, ear).nextz_idx;

    fn earcheck(ll: &LinkedLists, a: &Node, b: &Node, c: &Node, p: usize) -> bool {
        (p != a.idx)
            && (p != c.idx)
            && point_in_triangle(&a, &b, &c, &node!(ll, p))
            && area(&prev!(ll, p), &node!(ll, p), &next!(ll, p)) >= 0.0
    }

    while (p != NULL) && (node!(ll, p).z >= min_z) && (n != NULL) && (node!(ll, n).z <= max_z) {
        dlog!(18, "look for points inside the triangle in both directions");
        if earcheck(ll, &a, &b, &c, p) {
            return false;
        }
        p = node!(ll, p).prevz_idx;

        if earcheck(ll, &a, &b, &c, n) {
            return false;
        }
        n = node!(ll, n).nextz_idx;
    }

    while (p != NULL) && (node!(ll, p).z >= min_z) {
        dlog!(18, "look for remaining points in decreasing z-order");
        if earcheck(ll, &a, &b, &c, p) {
            return false;
        }
        p = node!(ll, p).prevz_idx;
    }

    while n != NULL && node!(ll, n).z <= max_z {
        dlog!(18, "look for remaining points in increasing z-order");
        if earcheck(ll, &a, &b, &c, n) {
            return false;
        }
        n = node!(ll, n).nextz_idx;
    }
    true
}


fn filter_points(ll: &mut LinkedLists, start: NodeIdx, mut end: NodeIdx) -> NodeIdx {
    dlog!(4, "fn filter_points, eliminate colinear or duplicate points");
    if start == NULL {
        dlog!(4, "fn filter points, start null");
        return start;
    }
    if end == NULL {
        end = start;
    }
    if end >= ll.nodes.len() || start >= ll.nodes.len() {
        return NULL;
    }

    let mut p = start;
    let mut again = false;


	// this loop "wastes" calculations by going over the same points multiple
	// times. however, altering the location of the 'end' node can disrupt
	// the algorithm of other code that calls the filter_points function.
    loop {
        again = false;
        if 	!node!(ll, p).steiner 
			&& ( equals(&node!(ll,p),&next!(ll,p)) || 
area(&prev!(ll, p), &node!(ll, p), &next!(ll, p)) == 0.0 )
        {
            ll.remove_node(p);
            end = node!(ll, p).prev_idx;
            p = end;
            if p == node!(ll, p).next_idx {
                break;
            }
            again = true;
        } else {
            p = node!(ll, p).next_idx;
        }
        if !again && p == end {
            break;
        }
    }


/*            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;
            end = node!(ll, end).prev_idx;*/
    dlog!(4, "fn filter points end {}", node!(ll, end).i);
    return end;
}

// create a circular doubly linked list from polygon points in the
// specified winding order
fn linked_list(
    data: &Vec<f64>,
    start: usize,
    end: usize,
    dim: usize,
    clockwise: bool,
) -> (LinkedLists, usize) {
    let mut ll: LinkedLists = LinkedLists::new();
    let lastidx = linked_list_add_contour(&mut ll, data, start, end, dim, clockwise);
    (ll, lastidx)
}

fn linked_list_add_contour(
    ll: &mut LinkedLists,
    data: &Vec<f64>,
    start: usize,
    end: usize,
    dim: usize,
    clockwise: bool,
) -> usize {
    if start > data.len() || end > data.len() || data.len() == 0 {
        return NULL;
    }
    let mut lastidx = NULL;
    if clockwise == (signed_area(&data, start, end, dim) > 0.0) {
        for i in (start..end).step_by(dim) {
            lastidx = ll.insert_node(i, data[i], data[i + 1], lastidx);
        }
    } else {
        for i in (start..=(end - dim)).rev().step_by(dim) {
            lastidx = ll.insert_node(i, data[i], data[i + 1], lastidx);
        }
    }

    if equals(&node!(ll, lastidx), &next!(ll, lastidx)) {
        ll.remove_node(lastidx);
        lastidx = node!(ll, lastidx).next_idx;;
    }
    return lastidx;
}

// z-order of a point given coords and inverse of the longer side of
// data bbox
fn zorder(xf: f64, yf: f64, minx: f64, miny: f64, invsize: f64) -> u32 {
    // coords are transformed into non-negative 15-bit integer range
    let mut x: u32 = 32767 * ((xf - minx) * invsize).round() as u32;
    let mut y: u32 = 32767 * ((yf - miny) * invsize).round() as u32;

    // todo ... big endian?
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    x | (y << 1)
}

// check if a point lies within a convex triangle
fn point_in_triangle(a: &Node, b: &Node, c: &Node, p: &Node) -> bool {
    ((c.x - p.x) * (a.y - p.y) - (a.x - p.x) * (c.y - p.y) >= 0.0)
        && ((a.x - p.x) * (b.y - p.y) - (b.x - p.x) * (a.y - p.y) >= 0.0)
        && ((b.x - p.x) * (c.y - p.y) - (c.x - p.x) * (b.y - p.y) >= 0.0)
}

pub fn earcut(data: &Vec<f64>, hole_indices: &Vec<usize>, mut dim: usize) -> Vec<usize> {
    if dim == 0 {
        dim = 2
    };
    let outer_len = match hole_indices.len() {
        0 => data.len(),
        _ => hole_indices[0] * dim,
    };

    let (mut ll, mut outer_node) = linked_list(data, 0, outer_len, dim, true);
    let mut triangles: Vec<usize> = Vec::new();
    if ll.nodes.len() == 0 {
        return triangles;
    }

    let (mut minx, mut miny, mut invsize) = (0.0, 0.0, 0.0);

    outer_node = eliminate_holes(&mut ll, data, hole_indices, outer_node, dim);

    // if the shape is not too simple, we'll use z-order curve hash
    // later; calculate polygon bbox
    if data.len() > 80 * dim || TESTHASH {
		let mut bb = BoundingBox::new(&node!(ll,outer_node));
		ll.iter_range(outer_node..outer_node).for_each(|n| bb.expand(n));
//		for n in ll.iter_range(outer_node..outer_node) {
//			bb.expand(n);
//		}
		minx = bb.minx;
		miny = bb.miny;
        invsize = calc_invsize(bb.minx, bb.miny, bb.maxx, bb.maxy);
    }

    // so basically, for data len < 80*dim, minx,miny are 0
    earcut_linked(
        &mut ll,
        outer_node,
        &mut triangles,
        dim,
        minx,
        miny,
        invsize,
        0,
    );

    return triangles;
}

// signed area of a parallelogram
fn area(p: &Node, q: &Node, r: &Node) -> f64 {
    (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
}

// check if two points are equal
fn equals(p1: &Node, p2: &Node) -> bool {
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
fn cure_local_intersections(
    ll: &mut LinkedLists,
    instart: NodeIdx,
    triangles: &mut Vec<NodeIdx>,
    dim: usize,
) -> NodeIdx {
    dlog!(
        4,
        "fn cure_local_intersections i:{},{:?},{}",
        node!(ll, instart).i,
        triangles,
        dim
    );
    let mut p = instart;
    let mut start = instart;
    loop {
        let a = node!(ll, p).prev_idx;
        let b = next!(ll, p).next_idx;

        dlog!(8, "a:{} b:{} p:{} pn:{}", a, b, p, node!(ll, p).next_idx);
        dlog!(8, "a==b?{}", equals(&node!(ll, a), &node!(ll, b)));
        dlog!(
            8,
            "isct a p pn b {}",
            pseudo_intersects(&node!(ll, a), &node!(ll, p), &next!(ll, p), &node!(ll, b))
        );
        dlog!(
            8,
            "locin a b {}",
            locally_inside(ll, &node!(ll, a), &node!(ll, b))
        );
        dlog!(
            8,
            "locin b a {}",
            locally_inside(ll, &node!(ll, b), &node!(ll, a))
        );

        if !equals(&node!(ll, a), &node!(ll, b))
            && pseudo_intersects(&node!(ll, a), &node!(ll, p), &next!(ll, p), &node!(ll, b))
            && locally_inside(ll, &node!(ll, a), &node!(ll, b))
            && locally_inside(ll, &node!(ll, b), &node!(ll, a))
        {
            triangles.push(node!(ll, a).i / dim);
            triangles.push(node!(ll, p).i / dim);
            triangles.push(node!(ll, b).i / dim);

            // remove two nodes involved
            ll.remove_node(p);
            let nidx = node!(ll, p).next_idx;
            ll.remove_node(nidx);

            start = node!(ll, b).idx;
            p = start;
        }
        p = node!(ll, p).next_idx;
        if p == start {
            break;
        }
    }

    return p;
}

// try splitting polygon into two and triangulate them independently
fn split_earcut(
    ll: &mut LinkedLists,
    start: NodeIdx,
    triangles: &mut Vec<NodeIdx>,
    dim: usize,
    minx: f64,
    miny: f64,
    invsize: f64,
) {
    dlog!(
        4,
        "fn split_earcut i:{} {:?} {} {} {} {}",
        node!(ll, start).i,
        triangles,
        dim,
        minx,
        miny,
        invsize
    );
    // look for a valid diagonal that divides the polygon into two
    let mut a = start;
    loop {
        let mut b = next!(ll, a).next_idx;
        while b != node!(ll, a).prev_idx {
            let test = is_valid_diagonal(ll, &node!(ll, a), &node!(ll, b));
            if node!(ll, a).i != node!(ll, b).i
                && is_valid_diagonal(ll, &node!(ll, a), &node!(ll, b))
            {
                // split the polygon in two by the diagonal
                let mut c = split_bridge_polygon(ll, a, b);

                // filter colinear points around the cuts
                let an = node!(ll, a).next_idx;
                let cn = node!(ll, c).next_idx;
                a = filter_points(ll, a, an);
                c = filter_points(ll, c, cn);

                // run earcut on each half
                earcut_linked(ll, a, triangles, dim, minx, miny, invsize, 0);
                earcut_linked(ll, c, triangles, dim, minx, miny, invsize, 0);
                return;
            }
            b = node!(ll, b).next_idx;
        }
        a = node!(ll, a).next_idx;
        if a == start {
            break;
        }
    }
}

// find a bridge between vertices that connects hole with an outer ring
// and and link it
fn eliminate_hole(ll: &mut LinkedLists, hole: NodeIdx, outer_node: NodeIdx) {
    dlog!(
        4,
        "fn eliminate_hole hole.i:{} outernode.i:{}",
        node!(ll, hole).i,
        node!(ll, outer_node).i
    );
    let test_node = find_hole_bridge(ll, &node!(ll, hole), outer_node);
    if test_node != NULL {
        let b = split_bridge_polygon(ll, test_node, hole);
        let bn = next!(ll, b).idx;
        filter_points(ll, b, bn);
    }
}

// David Eberly's algorithm for finding a bridge between hole and outer polygon
fn find_hole_bridge(ll: &LinkedLists, hole: &Node, outer_node: NodeIdx) -> NodeIdx {
    dlog!(
        4,
        "fn find_hole_bridge i:{} i:{}",
        hole.i,
        node!(ll, outer_node).i
    );
    if outer_node >= ll.nodes.len() {
        return NULL;
    }
    let mut p = outer_node;
    let hx = hole.x;
    let hy = hole.y;
    let mut qx: f64 = std::f64::NEG_INFINITY;
    let mut m: NodeIdx = NULL;

    // find a segment intersected by a ray from the hole's leftmost
    // point to the left; segment's endpoint with lesser x will be
    // potential connection point

    loop {
        let (px, py) = (node!(ll, p).x, node!(ll, p).y);
        if (hy <= py) && (hy >= next!(ll, p).y) && (next!(ll, p).y != py) {
            let x = px + (hy - py) * (next!(ll, p).x - px) / (next!(ll, p).y - py);

            if (x <= hx) && (x > qx) {
                qx = x;
                if x == hx {
                    if hy == py {
                        return p;
                    }
                    if hy == next!(ll, p).y {
                        return next!(ll, p).idx;
                    };
                }
                if px < next!(ll, p).x {
                    m = p
                } else {
                    m = next!(ll, p).idx
                };
            }
        }
        p = next!(ll, p).idx;
        if p == outer_node {
            break;
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

    let stop = m;
    let mx = node!(ll, m).x;
    let my = node!(ll, m).y;
    let mut tan_min = std::f64::INFINITY;
    let mut tan;
    //    let mut tan = 0.0;

    p = next!(ll, m).idx;

    while p != stop {
        let (px, py) = (node!(ll, p).x, node!(ll, p).y);
        let x1 = if hy < my { hx } else { qx };
        let x2 = if hy < my { qx } else { hx };

        let n1 = Node::new(0, x1, hy, 0);
        let mp = Node::new(0, mx, my, 0);
        let n2 = Node::new(0, x2, hy, 0);

        if (hx >= px) && (px >= mx) && (hx != px) && point_in_triangle(&n1, &mp, &n2, &node!(ll, p))
        {
            tan = (hy - py).abs() / (hx - px); // tangential

            if ((tan < tan_min) || ((tan == tan_min) && (px > node!(ll, m).x)))
                && locally_inside(ll, &node!(ll, p), &hole)
            {
                m = p;
                tan_min = tan;
            }
        }
        p = next!(ll, p).idx;
    }

    return m;
}

// check if a diagonal between two polygon nodes is valid (lies in
// polygon interior)
fn is_valid_diagonal(ll: &LinkedLists, a: &Node, b: &Node) -> bool {
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

fn pseudo_intersects(p1: &Node, q1: &Node, p2: &Node, q2: &Node) -> bool {
    if (equals(p1, p2) && equals(q1, q2)) || (equals(p1, q2) && equals(q1, p2)) {
        return true;
    }
    return (area(p1, q1, p2) > 0.0) != (area(p1, q1, q2) > 0.0)
        && (area(p2, q2, p1) > 0.0) != (area(p2, q2, q1) > 0.0);
}

// check if a polygon diagonal intersects any polygon segments
fn intersects_polygon(ll: &LinkedLists, a: &Node, b: &Node) -> bool {
    ll.iter_range(a.idx..a.idx).any(|p| {
        p.i != a.i
            && next!(ll, p.idx).i != a.i
            && p.i != b.i
            && next!(ll, p.idx).i != b.i
            && pseudo_intersects(&p, &next!(ll, p.idx), a, b)
    })
}

// check if a polygon diagonal is locally inside the polygon
fn locally_inside(ll: &LinkedLists, a: &Node, b: &Node) -> bool {
    match area(&prev!(ll, a.idx), a, &next!(ll, a.idx)) < 0.0 {
        true => area(a, b, &next!(ll, a.idx)) >= 0.0 && area(a, &prev!(ll, a.idx), b) >= 0.0,
        false => area(a, b, &prev!(ll, a.idx)) < 0.0 || area(a, &next!(ll, a.idx), b) < 0.0,
    }
}

// check if the middle point of a polygon diagonal is inside the polygon
fn middle_inside(ll: &LinkedLists, a: &Node, b: &Node) -> bool {
    let (mx, my) = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
    ll.iter_range(a.idx..a.idx).fold(false, |inside, p| {
        inside
            ^ (((p.y > my) != (next!(ll, p.idx).y > my))
                && (next!(ll, p.idx).y != p.y)
                && (mx
                    < ((next!(ll, p.idx).x - p.x) * (my - p.y) / (next!(ll, p.idx).y - p.y) + p.x)))
    })
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
fn split_bridge_polygon(ll: &mut LinkedLists, a: NodeIdx, b: NodeIdx) -> NodeIdx {
    dlog!(
        4,
        "fn split_bridge_polygon a.i:{} b.i:{}",
        node!(ll, a).i,
        node!(ll, b).i
    );
    let cidx = ll.nodes.len();
    let didx = cidx + 1;
    let mut c = Node::new(node!(ll, a).i, node!(ll, a).x, node!(ll, a).y, cidx);
    let mut d = Node::new(node!(ll, b).i, node!(ll, b).x, node!(ll, b).y, didx);

    let an = node!(ll, a).next_idx;
    let bp = node!(ll, b).prev_idx;

    node!(ll, a).next_idx = b;
    node!(ll, b).prev_idx = a;

    c.next_idx = an;
    node!(ll, an).prev_idx = cidx;

    d.next_idx = cidx;
    c.prev_idx = didx;

    node!(ll, bp).next_idx = didx;
    d.prev_idx = bp;

    ll.nodes.push(c);
    ll.nodes.push(d);
    return didx;
}

// return a percentage difference between the polygon area and its
// triangulation area; used to verify correctness of triangulation
pub fn deviation(
    data: &Vec<f64>,
    hole_indices: &Vec<usize>,
    dim: usize,
    triangles: &Vec<usize>,
) -> f64 {
    let mut indices = hole_indices.clone();
    indices.push(data.len() / dim);
    let (ix, iy) = (indices.iter(), indices.iter().skip(1));
    let body_area = signed_area(&data, 0, indices[0] * dim, dim).abs();
    let polygon_area = ix.zip(iy).fold(body_area, |a, (ix, iy)| {
        a - signed_area(&data, ix * dim, iy * dim, dim).abs()
    });

    let i = triangles.iter().skip(0).step_by(3).map(|x| x * dim);
    let j = triangles.iter().skip(1).step_by(3).map(|x| x * dim);
    let k = triangles.iter().skip(2).step_by(3).map(|x| x * dim);
    let triangles_area = i.zip(j).zip(k).fold(0., |ta, ((a, b), c)| {
        ta + ((data[a] - data[c]) * (data[b + 1] - data[a + 1])
            - (data[a] - data[b]) * (data[c + 1] - data[a + 1]))
            .abs()
    });
    match polygon_area == 0.0 && triangles_area == 0.0 {
        true => 0.0,
        false => ((triangles_area - polygon_area) / polygon_area).abs(),
    }
}

fn signed_area(data: &Vec<f64>, start: usize, end: usize, dim: usize) -> f64 {
    let i = (start..end).step_by(dim);
    let j = (start..end).cycle().skip((end - dim) - start).step_by(dim);
    i.zip(j).fold(0., |s, (i, j)| {
        s + (data[j] - data[i]) * (data[i + 1] + data[j + 1])
    })
}

// turn a polygon in a multi-dimensional array form (e.g. as in GeoJSON) 
// into a form Earcut accepts
pub fn flatten(data: &Vec<Vec<Vec<f64>>>) -> (Vec<f64>, Vec<usize>, usize) {
    (
        data.iter()
            .cloned()
            .flatten()
            .flatten()
            .collect::<Vec<f64>>(), // flat data
        data.iter()
            .take(data.len() - 1)
            .scan(0, |holeidx, v| {
                *holeidx += v.len();
                Some(*holeidx)
            }).collect::<Vec<usize>>(), // hole indexes
        data[0][0].len(), // dimensions
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
    fn dump(ll: &LinkedLists) -> String {
        let mut s = format!("LL, #nodes: {}", ll.nodes.len());
        s.push_str(&format!(" #used: {}\n", ll.nodes.len() - ll.freelist.len()));
        s.push_str(&format!(
            " {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2}\n",
            "vi", "i", "p", "n", "x", "y", "pz", "nz", "st", "fr", "cyl"
        ));
        for n in ll.nodes.iter() {
            s.push_str(&format!(
                " {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2}\n",
                n.idx,
                n.i,
                pn(n.prev_idx),
                pn(n.next_idx),
                n.x,
                n.y,
                pn(n.prevz_idx),
                pn(n.nextz_idx),
                pb(n.steiner),
                pb(ll.freelist.contains(&n.idx)),
                0//,ll.iter_range(n.idx..n.idx).count(),
            ));
        }
        return s;
    }


#[cfg(test)]
mod tests {
    use super::*;

    fn cycles_report(ll: &LinkedLists) -> String {
        if ll.nodes.len() == 0 {
            return format!("[]");
        }
        let mut markv: Vec<usize> = Vec::new();
        markv.resize(ll.nodes.len(), NULL);
        let mut cycler;;
        for i in 0..markv.len() {
            if ll.freelist.contains(&i) {
                markv[i] = NULL;
            } else if markv[i] == NULL {
                cycler = i;
                let mut p = i;
                let end = node!(ll, p).prev_idx;
                markv[p] = cycler;
                let mut count = 0;
                loop {
                    p = node!(ll, p).next_idx;
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

    fn cycle_dump(ll: &LinkedLists, p: NodeIdx) -> String {
        let mut s = format!("cycle from {}, ", p);
        s.push_str(&format!(" len {}, idxs:", cycle_len(&ll, p)));
        let mut i = p;
        let end = i;
        loop {
            s.push_str(&format!("{} ", node!(ll, i).idx));
            i = node!(ll, i).next_idx;
            if i == end {
                break s;
            }
        }
    }


    fn dump_cycle(ll: &LinkedLists, start: usize) -> String {
        let mut s = format!("LL, #nodes: {}", ll.nodes.len());
        s.push_str(&format!(" #used: {}\n", ll.nodes.len() - ll.freelist.len()));
        s.push_str(&format!(
            " {:>3} {:>3} {:>3} {:>4} {:>4} {:>8.3} {:>8.3} {:>4} {:>4} {:>2} {:>2} {:>2}\n",
            "#", "vi", "i", "p", "n", "x", "y", "pz", "nz", "st", "fr", "cyl"
        ));
        let mut startidx: usize = 0;
        for n in ll.nodes.iter() {
            if n.i == start {
                startidx = n.idx;
            };
        }
        let endidx = startidx;
        let mut idx = startidx;
        let mut count = 0;
        let mut state; // = 0u32;
        loop {
            let n = ll.nodes[idx].clone();
            state = 0; //horsh( state, n.i  as u32);
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
                pb(ll.freelist.contains(&n.idx)),
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

    fn cycle_len(ll: &LinkedLists, p: NodeIdx) -> usize {
        if p >= ll.nodes.len() {
            return 0;
        }
        let end = node!(ll, p).prev_idx;
        let mut i = p;
        let mut count = 1;
        loop {
            i = node!(ll, i).next_idx;
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
    fn horsh_ll(ll: &LinkedLists, starti: usize) -> String {
        let mut s = format!("LL horsh: ");
        let mut startidx: usize = 0;
        for n in ll.nodes.iter() {
            if n.i == starti {
                startidx = n.idx;
            };
        }
        let endidx = startidx;
        let mut idx = startidx;
        let mut count = 0;
        let mut state = 0u32;
        loop {
            let n = ll.nodes[idx].clone();
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
        let dims = 2;
        let data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&data, 0, data.len(), dims, true);
        assert!(ll.nodes.len() == 4);
        assert!(ll.nodes[0].idx == 0);
        assert!(ll.nodes[0].i == 6);
        assert!(ll.nodes[0].x == 1.0);
        assert!(ll.nodes[0].i == 6 && ll.nodes[0].y == 0.0);
        assert!(ll.nodes[0].next_idx == 1 && ll.nodes[0].prev_idx == 3);
        assert!(ll.nodes[3].next_idx == 0 && ll.nodes[3].prev_idx == 2);
        ll.remove_node(2);
    }

    #[test]
    fn test_point_in_triangle() {
        let dims = 2;
        let data = vec![0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.1];
        let (ll, _) = linked_list(&data, 0, data.len(), dims, true);
        assert!(point_in_triangle(
            &ll.nodes[0],
            &ll.nodes[1],
            &ll.nodes[2],
            &ll.nodes[3]
        ));
    }

    #[test]
    fn test_signed_area() {
        let data1 = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let data2 = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let a1 = signed_area(&data1, 0, 4, 2);
        let a2 = signed_area(&data2, 0, 4, 2);
        assert!(a1 == -a2);
    }

    #[test]
    fn test_deviation() {
        let data1 = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let tris = vec![0, 1, 2, 2, 3, 0];
        let hi: Vec<usize> = Vec::new();
        assert!(deviation(&data1, &hi, 2, &tris) == 0.0);
    }

    #[test]
    fn test_split_bridge_polygon() {
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let hole = vec![0.1, 0.1, 0.1, 0.2, 0.2, 0.2];
        body.extend(hole);
        let dims = 2;
        let (mut ll, _) = linked_list(&body, 0, body.len(), dims, true);
        assert!(cycle_len(&ll, 0) == body.len() / dims);
        let (left, right) = (0, 4);
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
        let dims = 2;

        let body = vec![0.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&body, 0, body.len(), dims, true);
        assert!(equals(&ll.nodes[0], &ll.nodes[1]));

        let body = vec![2.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&body, 0, body.len(), dims, true);
        assert!(!equals(&ll.nodes[0], &ll.nodes[1]));
    }

    #[test]
    fn test_area() {
        let dims = 2;
        let body = vec![4.0, 0.0, 4.0, 3.0, 0.0, 0.0]; // counterclockwise
        let (ll, _) = linked_list(&body, 0, body.len(), dims, true);
        assert!(area(&ll.nodes[0], &ll.nodes[1], &ll.nodes[2]) == -12.0);
        let body2 = vec![4.0, 0.0, 0.0, 0.0, 4.0, 3.0]; // clockwise
        let (ll2, _) = linked_list(&body2, 0, body2.len(), dims, true);
        // creation apparently modifies all winding to ccw
        assert!(area(&ll2.nodes[0], &ll2.nodes[1], &ll2.nodes[2]) == -12.0);
    }

    #[test]
    fn test_is_ear() {
        let dims = 2;
        let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dims, true);
        assert!(!is_ear(&ll, 0));
        assert!(!is_ear(&ll, 1));
        assert!(!is_ear(&ll, 2));

        let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.4];
        let (ll, _) = linked_list(&m, 0, m.len(), dims, true);
        assert!(is_ear(&ll, 0) == false);
        assert!(is_ear(&ll, 1) == true);
        assert!(is_ear(&ll, 2) == false);
        assert!(is_ear(&ll, 3) == true);

        let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dims, true);
        assert!(is_ear(&ll, 1));

        let m = vec![0.0, 0.0, 4.0, 0.0, 4.0, 3.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dims, true);
        assert!(is_ear(&ll, 1));
    }

    #[test]
    fn test_filter_points() {
        let dims = 2;
        let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), dims, true);
        let lllen = ll.nodes.len();
        let r1 = filter_points(&mut ll, 0, lllen - 1);
        assert!(cycle_len(&ll, r1) == 4);

        let n = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&n, 0, n.len(), dims, true);
        let lllen = ll.nodes.len();
        let r2 = filter_points(&mut ll, 0, lllen - 1);
        assert!(cycle_len(&ll, r2) == 4);

        let n2 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&n2, 0, n2.len(), dims, true);
        let r32 = filter_points(&mut ll, 0, 99);
        assert!(cycle_len(&ll, r32) != 4);

        let o = vec![0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5,0.5];
        let (mut ll, _) = linked_list(&o, 0, o.len(), dims, true);
        let lllen = ll.nodes.len();
        let r3 = filter_points(&mut ll, 0, lllen - 1);
        assert!(cycle_len(&ll, r3) == 3);


        let o = vec![0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&o, 0, o.len(), dims, true);
        let lllen = ll.nodes.len();
        let r3 = filter_points(&mut ll, 0, lllen - 1);
        assert!(cycle_len(&ll, r3) == 5);
    }

    #[test]
    fn test_earcut_linked() {
        let dim = 2;

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let (mut tris, minx, miny, invsize, pass) = (Vec::new(), 0.0, 0.0, 0.0, 0);
        earcut_linked(&mut ll, 0, &mut tris, dim, minx, miny, invsize, pass);
        assert!(tris.len() == 6);

        let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let (mut tris, minx, miny, invsize, pass) = (Vec::new(), 0.0, 0.0, 0.0, 0);
        earcut_linked(&mut ll, 0, &mut tris, dim, minx, miny, invsize, pass);
        assert!(tris.len() == 9);

        let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let (mut tris, minx, miny, invsize, pass) = (Vec::new(), 0.0, 0.0, 0.0, 0);
        earcut_linked(&mut ll, 0, &mut tris, dim, minx, miny, invsize, pass);
        assert!(tris.len() == 9);
    }

    #[test]
    fn test_middle_inside() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        assert!(middle_inside(&ll, &node!(ll, 0), &node!(ll, 2)));
        assert!(middle_inside(&ll, &node!(ll, 1), &node!(ll, 3)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        assert!(!middle_inside(&ll, &node!(ll, 0), &node!(ll, 2)));
        assert!(middle_inside(&ll, &node!(ll, 1), &node!(ll, 3)));
    }

    #[test]
    fn test_locally_inside() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 0)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 1)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 2)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 3)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 0)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 1)));
        assert!(!locally_inside(&ll, &node!(ll, 0), &node!(ll, 2)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 3)));
    }

    #[test]
    fn test_intersects_polygon() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);

        assert!(false == intersects_polygon(&ll, &node!(ll, 0), &node!(ll, 2)));
        assert!(false == intersects_polygon(&ll, &node!(ll, 2), &node!(ll, 0)));
        assert!(false == intersects_polygon(&ll, &node!(ll, 1), &node!(ll, 3)));
        assert!(false == intersects_polygon(&ll, &node!(ll, 3), &node!(ll, 1)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        dlog!(9, "{}", dump(&ll));
        dlog!(
            5,
            "{}",
            intersects_polygon(&ll, &node!(ll, 0), &node!(ll, 2))
        );
        dlog!(
            5,
            "{}",
            intersects_polygon(&ll, &node!(ll, 2), &node!(ll, 0))
        );
    }

    #[test]
    fn test_boundingbox() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 0.9, 0.9, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let mut bb = BoundingBox::new(&ll.nodes[0]);
        bb.expand(&ll.nodes[2]);
        assert!(bb.minx == 0.0 && bb.miny == 0.0 && bb.maxx == 0.9 && bb.maxy == 0.9);
        let mut bb = BoundingBox::new(&ll.nodes[1]);
        bb.expand(&ll.nodes[0]);
        assert!(bb.minx == 0.0 && bb.miny == 0.0 && bb.maxx == 1.0 && bb.maxy == 0.0);
    }

    #[test]
    fn test_intersects_itself() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 0.9, 0.9, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
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
        ti!(false, 0, 2, 0, 1);
        ti!(false, 0, 2, 1, 2);
        ti!(false, 0, 2, 2, 3);
        ti!(false, 0, 2, 3, 0);
        ti!(true, 0, 2, 3, 1);
        ti!(true, 0, 2, 1, 3);
        ti!(true, 2, 0, 3, 1);
        ti!(true, 2, 0, 1, 3);
        ti!(false, 0, 1, 2, 3);
        ti!(false, 1, 0, 2, 3);
        ti!(false, 0, 0, 2, 3);
        ti!(false, 0, 1, 3, 2);
        ti!(false, 1, 0, 3, 2);

        ti!(true, 0, 2, 2, 0); // special cases
        ti!(true, 0, 2, 0, 2);

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        assert!(false == pseudo_intersects(&ll.nodes[3], &ll.nodes[4], &ll.nodes[0], &ll.nodes[2]));

        // special case
        assert!(true == pseudo_intersects(&ll.nodes[3], &ll.nodes[4], &ll.nodes[2], &ll.nodes[0]));
    }

    #[test]
    fn test_is_valid_diagonal() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        assert!(!is_valid_diagonal(&ll, &ll.nodes[0], &ll.nodes[1]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[1], &ll.nodes[2]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[2], &ll.nodes[3]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[3], &ll.nodes[0]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[0], &ll.nodes[2]));
        assert!(is_valid_diagonal(&ll, &ll.nodes[1], &ll.nodes[3]));
        assert!(!is_valid_diagonal(&ll, &ll.nodes[2], &ll.nodes[0]));
        assert!(is_valid_diagonal(&ll, &ll.nodes[3], &ll.nodes[1]));
    }

    #[test]
    fn test_find_hole_bridge() {
        let dim = 2;

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let hole = Node::new(0, 0.8, 0.8, NULL);
        assert!(0 == find_hole_bridge(&ll, &hole, 0));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.4, 0.5];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let hole = Node::new(0, 0.5, 0.5, NULL);
        assert!(4 == find_hole_bridge(&ll, &hole, 0));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -0.4, 0.5];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let hole = Node::new(0, 0.5, 0.5, NULL);
        assert!(4 == find_hole_bridge(&ll, &hole, 0));

        let m = vec![
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -0.1, 0.9, 0.1, 0.8, -0.1, 0.7, 0.1, 0.6, -0.1,
            0.5,
        ];
        let (ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let hole = Node::new(0, 0.5, 0.9, NULL);
        assert!(4 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.1, NULL);
        assert!(8 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.5, NULL);
        assert!(8 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.55, NULL);
        assert!(8 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.6, NULL);
        assert!(7 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.65, NULL);
        assert!(6 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.7, NULL);
        assert!(6 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.75, NULL);
        assert!(6 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.8, NULL);
        assert!(5 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.85, NULL);
        assert!(4 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.9, NULL);
        assert!(4 == find_hole_bridge(&ll, &hole, 0));
        let hole = Node::new(0, 0.2, 0.95, NULL);
        assert!(4 == find_hole_bridge(&ll, &hole, 0));
    }

    #[test]
    fn test_eliminate_hole() {
        let dims = 2;
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

        let hole = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        let (mut ll, _) = linked_list(&body, 0, bodyend, dims, true);
        linked_list_add_contour(&mut ll, &body, holestart, holeend, dims, false);
        assert!(cycle_len(&ll, 0) == 4);
        assert!(cycle_len(&ll, 5) == 4);
        eliminate_hole(&mut ll, holestart / dims, 0);
        assert!(cycle_len(&ll, 0) == 10);

        let hole = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        linked_list_add_contour(&mut ll, &body, holestart, holeend, dims, false);
        assert!(cycle_len(&ll, 0) == 10);
        assert!(cycle_len(&ll, 5) == 10);
        assert!(cycle_len(&ll, 10) == 4);
        eliminate_hole(&mut ll, 10, 0);
        assert!(!cycle_len(&ll, 0) != 10);
        assert!(!cycle_len(&ll, 0) != 10);
        assert!(!cycle_len(&ll, 5) != 10);
        assert!(!cycle_len(&ll, 10) != 4);
        assert!(cycle_len(&ll, 0) == 16);
        assert!(cycle_len(&ll, 1) == 16);
        assert!(cycle_len(&ll, 10) == 16);
        assert!(cycle_len(&ll, 15) == 16);
    }

    #[test]
    fn test_cycle_len() {
        let dims = 2;
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.1, 0.1];

        let hole = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        let (mut ll, _) = linked_list(&body, 0, bodyend, dims, true);
        linked_list_add_contour(&mut ll, &body, holestart, holeend, dims, false);

        let hole = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        linked_list_add_contour(&mut ll, &body, holestart, holeend, dims, false);

        dlog!(5, "{}", dump(&ll));
        dlog!(5, "{}", cycles_report(&ll));
    }

    #[test]
    fn test_cycles_report() {
        let dims = 2;
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.1, 0.1];

        let hole = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        let (mut ll, _) = linked_list(&body, 0, bodyend, dims, true);
        linked_list_add_contour(&mut ll, &body, holestart, holeend, dims, false);

        let hole = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8];
        let bodyend = body.len();
        body.extend(hole);
        let holestart = bodyend;
        let holeend = body.len();
        linked_list_add_contour(&mut ll, &body, holestart, holeend, dims, false);

        dlog!(5, "{}", dump(&ll));
        dlog!(5, "{}", cycles_report(&ll));
    }

    #[test]
    fn test_eliminate_holes() {
        let dims = 2;
        let mut hole_indices: Vec<usize> = Vec::new();
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (mut ll, _) = linked_list(&body, 0, body.len(), dims, true);
        let hole1 = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
        let hole2 = vec![0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8];
        hole_indices.push(body.len() / dims);
        hole_indices.push((body.len() + hole1.len()) / dims);
        body.extend(hole1);
        body.extend(hole2);

        eliminate_holes(&mut ll, &body, &hole_indices, 0, 2);

		for i in 0..13 {
			if !ll.freelist.contains(&i) {
		        assert!(cycle_len(&ll, i)== body.len() / 2 + 2 + 2 );
			}
		}
    }

    #[test]
    fn test_cure_local_intersections() {
        let dim = 2;
        // first test - it would be nice if it "detected" this but
        // the points are not 'local' enough to each other in the cycle
        let m = vec![
            0.0, 0.0, 1.0, 0.0, 1.1, 0.1, 0.9, 0.1, 1.0, 0.05, 1.0, 1.0, 0.0, 1.0,
        ];
        let (mut ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let mut triangles: Vec<usize> = Vec::new();
        cure_local_intersections(&mut ll, 0, &mut triangles, dim);
        assert!(cycle_len(&ll, 0) == 7);
        assert!(ll.freelist.len() == 0);
        assert!(triangles.len() == 0);

        // second test - we have three points that immediately cause
        // self intersection. so it should, in theory, detect and clean
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.1, 0.1, 1.1, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (mut ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let mut triangles: Vec<usize> = Vec::new();
        cure_local_intersections(&mut ll, 0, &mut triangles, dim);
        assert!(cycle_len(&ll, 0) == 4);
        assert!(ll.freelist.len() == 2);
        assert!(triangles.len() == 3);
    }

    #[test]
    fn test_split_earcut() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

        let (minx, miny, maxx, maxy) = (0.0, 0.0, 1.0, 1.0);
        let invsize = calc_invsize(minx, miny, maxx, maxy);
        let dim = 2;
        let (mut ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let start = 0;
        let mut triangles: Vec<usize> = Vec::new();
        split_earcut(&mut ll, start, &mut triangles, dim, minx, miny, invsize);
        assert!(triangles.len() == 6);
        assert!(ll.nodes.len() == 6);
        assert!(ll.freelist.len() == 2);

        let m = vec![
            0.0, 0.0, 1.0, 0.0, 1.5, 0.5, 2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0, 1.5, 0.6, 1.0,
            1.0, 0.0, 1.0,
        ];
        let (minx, miny, maxx, maxy) = (0.0, 0.0, 1.0, 1.0);
        let invsize = calc_invsize(minx, miny, maxx, maxy);
        let dim = 2;
        let (mut ll, _) = linked_list(&m, 0, m.len(), dim, true);
        let start = 0;
        let mut triangles: Vec<usize> = Vec::new();
        split_earcut(&mut ll, start, &mut triangles, dim, minx, miny, invsize);
        assert!(ll.nodes.len() == 12);
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
        let (coords, hole_indices, dim) = flatten(&data);
        println!("{:?} {:?} {:?}", coords, hole_indices, dim);
        assert!(coords.len() == 24);
        assert!(hole_indices.len() == 2);
        assert!(hole_indices[0] == 4);
        assert!(hole_indices[1] == 8);
        assert!(dim == 2);
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
        let (coords, hole_indices, dim) = flatten(&data);
        let triangles = earcut(&coords, &hole_indices, dim);
        assert!(triangles.len() > 4);
    }
}

