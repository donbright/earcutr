#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_macros)]

static NULL: usize = 0x777A91CC;
static DEBUG: usize = 2;

type NodeIdx = usize;

impl Node {
    fn new(i: usize, x: f32, y: f32, idx: usize) -> Node {
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

#[derive(Clone)]
struct Node {
    i: usize,        // vertex index in coordinates array
    x: f32,          // vertex x coordinates
    y: f32,          // vertex y coordinates
    prev_idx: usize, // previous vertex nodes in a polygon ring
    next_idx: usize,
    z: u32,           // z-order curve value
    prevz_idx: usize, // previous and next nodes in z-order
    nextz_idx: usize,
    steiner: bool, // indicates whether this is a steiner point
    idx: usize,    // index within vector that holds all nodes
}

macro_rules! node {
    ($ll:ident,$idx:expr) => {
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
macro_rules! nextz {
    ($ll:ident,$idx:expr) => {
        $ll.nodes[$ll.nodes[$idx].nextz_idx]
    };
}
macro_rules! prevz {
    ($ll:ident,$idx:expr) => {
        $ll.nodes[$ll.nodes[$idx].prevz_idx]
    };
}

struct LL {
    nodes: Vec<Node>,
    headlist: Vec<NodeIdx>,
    freelist: Vec<NodeIdx>,
}

impl LL {
    fn cycle_dump(&self, p: NodeIdx) -> String {
        let mut s = format!("cycle from {}, ", p);
        s.push_str(&format!(" len {}, idxs:", self.cycle_len(p)));
        let mut i = p;
        let end = i;
        loop {
            s.push_str(&format!("{} ", node!(self, i).idx));
            i = node!(self, i).next_idx;
            if i == end {
                break s;
            }
        }
    }
    fn dump(&self) -> String {
        fn pn(a: usize) -> String {
            if a == NULL {
                return String::from("NULL");
            } else {
                return a.to_string();
            }
        }
        let mut s = format!("LL, num of nodes: {}\n", self.nodes.len());
        s.push_str(&format!(
            " {:>3} {:>3} {:>4} {:>4} {:>6} {:>6} {:>4} {:>4} {:>4}\n",
            "vi", "i", "p", "n", "x", "y", "pz", "nz", "st"
        ));
        for (vi, n) in self.nodes.iter().enumerate() {
            s.push_str(&format!(
                " {:>3} {:>3} {:>4} {:>4} {:>6} {:>6} {:>4} {:>4} {:>4}\n",
                n.idx,
                n.i,
                pn(n.prev_idx),
                pn(n.next_idx),
                n.x,
                n.y,
                pn(n.prevz_idx),
                pn(n.nextz_idx),
                n.steiner,
            ));
        }
        s.push_str(&format!("freelist: {:?} ", self.freelist));
        s.push_str(&format!("headlist: {:?}", self.headlist));
        return s;
    }
    fn insert_node(&mut self, i: usize, x: f32, y: f32) {
        let ll = self;
        let ls = ll.nodes.len();
        let mut p = Node::new(i, x, y, ls);
        if ls == 0 {
            p.next_idx = 0;
            p.prev_idx = 0;
            ll.headlist.push(0);
        } else {
            p.next_idx = ll.nodes[ls - 1].next_idx;
            p.prev_idx = ls - 1;
            let z = ll.nodes[ls - 1].next_idx;
            ll.nodes[z].prev_idx = ls;
            ll.nodes[ls - 1].next_idx = ls;
        }
        ll.nodes.push(p.clone());
    }
    fn cycle_len(&self, p: NodeIdx) -> usize {
        if p >= self.nodes.len() {
            return 0;
        }
        let end = node!(self, p).prev_idx;
        let mut i = p;
        let mut count = 1;
        loop {
            i = node!(self, i).next_idx;
            count += 1;
            if i == end {
                break count;
            }
        }
    }
    fn remove_node(&mut self, p: NodeIdx) {
        self.freelist.push(p);

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
    }
    fn new() -> LL {
        LL {
            nodes: Vec::new(),
            headlist: Vec::new(),
            freelist: Vec::new(),
        }
    }
    // link every hole into the outer loop, producing a single-ring polygon
    // without holes
    fn eliminate_holes(
        &mut self,
        data: &Vec<f32>,
        hole_indices: &Vec<usize>,
        outer_node: NodeIdx,
        dim: usize,
    ) -> NodeIdx {
        return outer_node;
        /*    let queue = [],
    let (i, len, start, end, list) 

    for (i = 0, len = hole_indices.len(); i < len; i+=1) {
        start = hole_indices[i] * dim;
        end = i < len - 1 ? hole_indices[i + 1] * dim : data.len();
        list = linkedList(data, start, end, dim:usize, false);
        if (list == list.next_idx) list.steiner = true;
        queue.push(get_leftmost(list));
    }

    queue.sort(compareX);

    // process holes from left to right
    for (i = 0; i < queue.len(); i+=1) {
        eliminateHole(queue[i], outerNode);
        outerNode = filter_points(outerNode, outerNode.next_idx);
    }

    return outerNode;
*/
    } // elim holes
} // LL

// minx, miny and invsize are later used to transform coords
// into integers for z-order calculation
fn calc_invsize(minx: f32, miny: f32, maxx: f32, maxy: f32) -> f32 {
    let mut invsize = maxf(maxx - minx, maxy - miny);
    if invsize != 0.0 {
        invsize = 1.0 / invsize;
    } else {
        invsize = 0.0;
    }
    invsize
}

// main ear slicing loop which triangulates a polygon (given as a linked
// list)
fn earcut_linked(
    ll: &mut LL,
    mut ear: NodeIdx,
    triangles: &mut Vec<usize>,
    dim: usize,
    minx: f32,
    miny: f32,
    invsize: f32,
    pass: usize,
) {
    if DEBUG > 4 {
        println!(
            "earcut_linked nodes:{} tris:{} dm:{} mx:{} my:{} invs:{} pas:{}",
            ll.nodes.len(),
            triangles.len(),
            dim,
            minx,
            miny,
            invsize,
            pass
        );
    }
    if ear == NULL {
        return;
    }

    // interlink polygon nodes in z-order
    // note this does nothing for smaller data len, b/c invsize will be 0
    if pass == 0 && invsize > 0.0 {
        index_curve(ll, ear, minx, miny, invsize);
    }

    let mut stop = ear;
    let mut prev = 0;
    let mut next = 0;
    // iterate through ears, slicing them one by one
    while node!(ll, ear).prev_idx != node!(ll, ear).next_idx {
        if DEBUG > 4 {
            println!("p{} e{} n{} s{}", prev, ear, next, stop);
            ll.dump();
        }
        prev = node!(ll, ear).prev_idx;
        next = node!(ll, ear).next_idx;

        let test;
        if invsize > 0.0 {
            test = is_ear_hashed(ll, ear, minx, miny, invsize);
        } else {
            test = is_ear(ll, ear);
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

        // f we looped through the whole remaining polygon and can't
        // find any more ears
        if ear == stop {
            /*
                        // try filtering points and slicing again
//            if (!pass) { earcut_linked(filter_points(ear), triangles, 
//                dim:usize, minx, miny, invsize, 1);
            // if this didn't work, try curing all small 
            // self-intersections locally
            } 
*/
            /*else if (pass == 1) {
                ear = cureLocalIntersections(ear, triangles, dim:usize); 
                earcut_linked(ear, triangles, dim:usize, minx, miny, 
                invsize, 2);
            // as a last resort, try splitting the remaining polygon 
            // into two
            } else if (pass == 2) {
                splitEarcut(ear, triangles, dim:usize, minx, miny, invsize);
            }
  */
            break;
        }
    } // while
    if DEBUG > 1 {
        println!("earcut_linked end");
    }
} //cut_linked

// interlink polygon nodes in z-order
fn index_curve(ll: &mut LL, start: usize, minx: f32, miny: f32, invsize: f32) {
    if DEBUG > 1 {
        println!("index curve");
    }
    let mut nodeidx = start;
    loop {
        let p = &mut ll.nodes[nodeidx]; //node!(ll,nodeidx);
        if p.z == NULL as u32 {
            p.z = zorder(p.x, p.y, minx, miny, invsize);
        }
        p.prevz_idx = p.prev_idx;
        p.nextz_idx = p.next_idx;
        nodeidx = p.next_idx;
        if nodeidx == start {
            break;
        }
    }

    let pzidx = node!(ll, nodeidx).prevz_idx;
    node!(ll, pzidx).nextz_idx = NULL;
    node!(ll, nodeidx).prevz_idx = NULL;

    sort_linked(ll);
    if DEBUG > 1 {
        println!("index curve end");
    }
} // indexcurve

// Simon Tatham's linked list merge sort algorithm
// http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
fn sort_linked(ll: &mut LL) {
    if DEBUG > 1 {
        println!("sort linked");
    }
    let (i, tailidx) = (0, 0);
    let mut nummerges;
    let mut psize;
    let mut qsize;
    let mut pidx;
    let mut qidx;
    let mut eidx;
    let mut insize = 1;
    loop {
        pidx = 0;
        let mut listidx = 0;
        let mut tailidx = 0;
        nummerges = 0;

        while pidx != NULL {
            nummerges += 1;
            qidx = pidx;
            psize = 0;
            for i in 0..insize {
                psize += 1;
                qidx = node!(ll, qidx).nextz_idx;
                if qidx == NULL {
                    break;
                }
            }
            qsize = insize;

            while psize > 0 || (qsize > 0 && qidx != NULL) {
                if psize != 0
                    && (qsize == 0 || qidx == NULL || ll.nodes[pidx].z <= ll.nodes[qidx].z)
                {
                    eidx = pidx;
                    pidx = ll.nodes[pidx].nextz_idx;
                    psize -= 1;
                } else {
                    eidx = qidx;
                    qidx = ll.nodes[qidx].nextz_idx;
                    qsize -= 1;
                }

                if tailidx != NULL {
                    ll.nodes[tailidx].nextz_idx = eidx;
                } else {
                    listidx = eidx;
                }

                ll.nodes[eidx].prevz_idx = tailidx;
                tailidx = eidx;
            }

            pidx = qidx;
        }

        ll.nodes[tailidx].nextz_idx = NULL;
        insize *= 2;
        if nummerges <= 1 {
            break;
        }
    } // while (nummerges > 1);
    if DEBUG > 1 {
        println!("sort linked end");
        if DEBUG > 4 {
            println!("{}", ll.dump());
        }
    }
} // end sort

// check whether a polygon node forms a valid ear with adjacent nodes
fn is_ear(ll: &LL, ear: usize) -> bool {
    if ear > ll.nodes.len() {
        return false;
    }
    if DEBUG > 2 {
        println!("is ear, {}", ear);
    }
    let result = true;
    let a = &prev!(ll, ear);
    let b = &node!(ll, ear);
    let c = &next!(ll, ear);

    // reflex, can't be an ear
    if area(a, b, c) >= 0.0 {
        return false;
    }

    // now make sure we don't have other points inside the potential ear
    let mut p = c.next_idx;
    while p != a.idx {
        if point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, node!(ll, p).x, node!(ll, p).y)
            && (area(&prev!(ll, p), &node!(ll, p), &next!(ll, p)) >= 0.0)
        {
            return false;
        }
        p = next!(ll, p).idx;
    }
    true
}

fn is_ear_hashed(ll: &mut LL, ear: usize, minx: f32, miny: f32, invsize: f32) -> bool {
    //    invsize;
    return false;
} // is ear hashed

/*
    let a = ear.prev_idx,
        b = ear,
        c = ear.next_idx;

    if (area(a, b, c) >= 0) return false; // reflex, can't be an ear

    // triangle bbox; 
min & max are calculated like this for speed
    let minTX = a.x < b.x ? (a.x < c.x ? a.x : c.x) : (b.x < c.x ? b.x : c.x),
        minTY = a.y < b.y ? (a.y < c.y ? a.y : c.y) : (b.y < c.y ? b.y : c.y),
        maxTX = a.x > b.x ? (a.x > c.x ? a.x : c.x) : (b.x > c.x ? b.x : c.x),
        maxTY = a.y > b.y ? (a.y > c.y ? a.y : c.y) : (b.y > c.y ? b.y : c.y);

    // z-order range for the current triangle bbox;
    let minZ = zorder(minTX, minTY, minx, miny, invsize),
        maxZ = zorder(maxTX, maxTY, minx, miny, invsize);

    let p = ear.prevz_idx,
        n = ear.nextz_idx;

    // look for points inside the triangle in both directions
    while (p && p.z >= minZ && n && n.z <= maxZ) {
        if (p != ear.prev_idx && p != ear.next_idx &&
            point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) &&
            area(p.prev_idx, p, p.next_idx) >= 0) return false;
        p = p.prevz_idx;

        if (n != ear.prev_idx && n != ear.next_idx &&
            point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, n.x, n.y) &&
            area(n.prev_idx, n, n.next_idx) >= 0) return false;
        n = n.nextz_idx;
    }

    // look for remaining points in decreasing z-order
    while (p && p.z >= minZ) {
        if (p != ear.prev_idx && p != ear.next_idx &&
            point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) &&
            area(p.prev_idx, p, p.next_idx) >= 0) return false;
        p = p.prevz_idx;
    }

    // look for remaining points in increasing z-order
    while (n && n.z <= maxZ) {
        if (n != ear.prev_idx && n != ear.next_idx &&
            point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, n.x, n.y) &&
            area(n.prev_idx, n, n.next_idx) >= 0) return false;
        n = n.nextz_idx;
    }
    return true;
*/
// eliminate colinear or duplicate points
fn filter_points(ll: &mut LL, start: NodeIdx, mut end: NodeIdx) -> NodeIdx {
    if start == NULL {
        return start;
    }
    if end == NULL {
        end = start;
    }

    if end >= ll.nodes.len() || start >= ll.nodes.len() {
        //println!("filter problem, {} {} {}",start,end,ll.nodes.len());
        return NULL;
    }

    let mut p = start;
    let mut again;
    loop {
        again = false;
        if (!(node!(ll, p).steiner))
            && (equals(&node!(ll, p), &next!(ll, p))
                || area(&prev!(ll, p), &node!(ll, p), &next!(ll, p)) == 0.0)
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
        if !(again || p != end) {
            break;
        }
    }

    return end;
} //filter

// create a circular doubly linked list from polygon points in the
// specified winding order
fn linked_list(data: &Vec<f32>, start: usize, end: usize, dim: usize, clockwise: bool) -> LL {
    let mut ll: LL = LL::new();
    if start > data.len() || end > data.len() {
        return ll;
    }
    if clockwise == (signed_area(&data, start, end, dim) > 0.0) {
        for i in (start..end).step_by(dim) {
            ll.insert_node(i, data[i], data[i + 1]);
        }
    } else {
        for i in (start..=(end - dim)).rev().step_by(dim) {
            ll.insert_node(i, data[i], data[i + 1]);
        }
    }
    // todo, remove duplicate point at end of list
    return ll;
}

// z-order of a point given coords and inverse of the longer side of
// data bbox
fn zorder(xf: f32, yf: f32, minx: f32, miny: f32, invsize: f32) -> u32 {
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
fn point_in_triangle(
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    cx: f32,
    cy: f32,
    px: f32,
    py: f32,
) -> bool {
    //println!("pt in ti {},{} {},{} {},{} ({},{})",ax,ay,bx,by,cx,cy,px,py);
    let r = ((cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0.0)
        && ((ax - px) * (by - py) - (bx - px) * (ay - py) >= 0.0)
        && ((bx - px) * (cy - py) - (cx - px) * (by - py) >= 0.0);
    //println!("pt in ti res {}",r);
    return r;
}

// return greater of two floating point numbers
fn maxf(a: f32, b: f32) -> f32 {
    if a > b {
        return a;
    }
    return b;
}

pub fn earcut(data: &Vec<f32>, hole_indices: &Vec<usize>, ndim: usize) -> Vec<usize> {
    if DEBUG > 4 {
        println!("earcut");
    }
    let mut dim = ndim;
    if dim == 0 {
        dim = 2
    };
    let has_holes = hole_indices.len() > 0;
    let mut outer_len = data.len();
    if has_holes {
        outer_len = hole_indices[0] * dim;
    }
    let mut ll = linked_list(data, 0, outer_len, dim, true);
    let mut outer_node = ll.nodes.len() - 1;
    let mut triangles: Vec<usize> = Vec::new();
    if outer_node == 0 {
        if DEBUG > 4 {
            println!("no nodes, triangles: {:?}", triangles);
        }
        return triangles;
    }

    let (mut minx, mut miny, mut invsize) = (0.0, 0.0, 0.0);
    let (mut maxx, mut maxy, mut x, mut y);

    if has_holes {
        outer_node = ll.eliminate_holes(data, hole_indices, outer_node, dim);
    }

    // if the shape is not too simple, we'll use z-order curve hash
    // later; calculate polygon bbox
    if DEBUG > 4 {
        println!(" data len {}", data.len());
    }
    if data.len() > 80 * dim {
        minx = data[0];
        maxx = data[0];
        miny = data[1];
        maxy = data[1];

        for i in (dim..outer_len).step_by(dim) {
            x = data[i];
            y = data[i + 1];
            if x < minx {
                minx = x;
            }
            if y < miny {
                miny = y;
            }
            if x > maxx {
                maxx = x;
            }
            if y > maxy {
                maxy = y;
            }
        }

        invsize = calc_invsize(minx, miny, maxx, maxy);
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

    if DEBUG > 4 {
        println!("earcut end");
    }
    if DEBUG > 4 {
        println!("triangles: {:?}", triangles);
    }
    return triangles;
}

// signed area of a parallelogram
fn area(p: &Node, q: &Node, r: &Node) -> f32 {
    (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
}

// check if two points are equal
fn equals(p1: &Node, p2: &Node) -> bool {
    p1.x == p2.x && p1.y == p2.y
}

/*
// go through all polygon nodes and cure small local self-intersections
fn cureLocalIntersections(start, triangles, dim:usize) {
    let p = start;
    do {
        let a = p.prev_idx,
            b = p.next_idx.next_idx;

        if (!equals(a, b) && intersects(a, p, p.next_idx, b) && locally_inside(a, b) && locally_inside(b, a)) {

            triangles.push(a.i / dim);
            triangles.push(p.i / dim);
            triangles.push(b.i / dim);

            // remove two nodes involved
            ll.remove_node(p);
            ll.remove_node(p.next_idx);

            p = start = b;
        }
        p = p.next_idx;
    } while (p != start);

    return p;
}

// try splitting polygon into two and triangulate them independently
fn splitEarcut(start, triangles, dim:usize, minx, miny, invsize) {
    // look for a valid diagonal that divides the polygon into two
    let a = start;
    do {
        let b = a.next_idx.next_idx;
        while (b != a.prev_idx) {
            if (a.i != b.i && is_valid_diagonal(a, b)) {
                // split the polygon in two by the diagonal
                let c = split_polygon(a, b);

                // filter colinear points around the cuts
                a = filter_points(a, a.next_idx);
                c = filter_points(c, c.next_idx);

                // run earcut on each half
                earcut_linked(a, triangles, dim:usize, minx, miny, invsize);
                earcut_linked(c, triangles, dim:usize, minx, miny, invsize);
                return;
            }
            b = b.next_idx;
        }
        a = a.next_idx;
    } while (a != start);
}

fn compareX(a, b) {
    return a.x - b.x;
}

// find a bridge between vertices that connects hole with an outer ring and and link it
fn eliminateHole(hole, outerNode) {
    outerNode = findHoleBridge(hole, outerNode);
    if (outerNode) {
        let b = split_polygon(outerNode, hole);
        filter_points(b, b.next_idx);
    }
}

// David Eberly's algorithm for finding a bridge between hole and outer polygon
fn findHoleBridge(hole, outerNode) {
    let p = outerNode,
        hx = hole.x,
        hy = hole.y,
        qx = -Infinity,
        m;

    // find a segment intersected by a ray from the hole's leftmost point to the left;
    // segment's endpoint with lesser x will be potential connection point
    do {
        if (hy <= p.y && hy >= p.next_idx.y && p.next_idx.y != p.y) {
            let x = p.x + (hy - p.y) * (p.next_idx.x - p.x) / (p.next_idx.y - p.y);
            if (x <= hx && x > qx) {
                qx = x;
                if (x == hx) {
                    if (hy == p.y) return p;
                    if (hy == p.next_idx.y) return p.next_idx;
                }
                m = p.x < p.next_idx.x ? p : p.next_idx;
            }
        }
        p = p.next_idx;
    } while (p != outerNode);

    if (!m) return NULL;

    if (hx == qx) return m.prev_idx; // hole touches outer segment; pick lower endpoint

    // look for points inside the triangle of hole point, segment intersection and endpoint;
    // if there are no points found, we have a valid connection;
    // otherwise choose the point of the minimum angle with the ray as connection point

    let stop = m,
        mx = m.x,
        my = m.y,
        tanMin = Infinity,
        tan;

    p = m.next_idx;

    while (p != stop) {
        if (hx >= p.x && p.x >= mx && hx != p.x &&
                point_in_triangle(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p.x, p.y)) {

            tan = Math.abs(hy - p.y) / (hx - p.x); // tangential

            if ((tan < tanMin || (tan == tanMin && p.x > m.x)) && locally_inside(p, hole)) {
                m = p;
                tanMin = tan;
            }
        }

        p = p.next_idx;
    }

    return m;
}
*/

// get the leftmost node in the list, given a starting node
fn get_leftmost(ll: &LL, start: NodeIdx) -> NodeIdx {
    let mut p = start;
    let mut leftmost = start;
    while {
        if node!(ll, p).x < node!(ll, leftmost).x {
            leftmost = p
        };
        p = node!(ll, p).next_idx;
        p != start
    } {}
    return leftmost;
}

// check if a diagonal between two polygon nodes is valid (lies in
// polygon interior)
fn is_valid_diagonal(ll: &LL, a: &Node, b: &Node) -> bool {
    return next!(ll, a.idx).i != b.i
        && prev!(ll, a.idx).i != b.i
        && !intersects_polygon(ll, a, b)
        && locally_inside(ll, a, b)
        && locally_inside(ll, b, a)
        && middle_inside(ll, a, b);
}

// does the axis-aligned bounding box defined by points a and b intersect
// with the box defined by points c and d? 
fn bbox_intersect(a: &Node, b: &Node, c: &Node, d: &Node) -> bool {
    ( (a.x - c.x)*(b.x - c.x)<=0.0 || (a.x - d.x)*(b.x - d.x)<=0.0 ) 
  && ( (a.y - c.y)*(b.y - c.y)<=0.0 || (a.y - d.y)*(b.y - d.y)<=0.0 )
}

/* check if two segments intersect, line segment p1-q1 vs line segment p2-q2
bsed on https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

this has been modified from the version in earcut.js for the case where
endpoints touch each other.

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
fn intersects(p1: &Node, q1: &Node, p2: &Node, q2: &Node) -> bool {
    if (equals(p1, p2) || equals(q1, q2)) || (equals(p1, q2) || equals(p2, q1)) {
        return true; // points touch
    };
    let a1 = area(p1, q1, p2);
    let a2 = area(p1, q1, q2);
    let a3 = area(p2, q2, p1);
    let a4 = area(p2, q2, q1);
    if (a1 == 0.0 && a2 == 0.0) || (a3 == 0.0 && a4 == 0.0) {
		// collinear
        return bbox_intersect(p1, q1, p2, q2);
    }
	// check windings
    let r = ((a1 > 0.0) != (a2 > 0.0)) && ((a3 > 0.0) != (a4 > 0.0));
    return r;
}

// check if a polygon diagonal intersects any polygon segments
fn intersects_polygon(ll: &LL, a: &Node, b: &Node) -> bool {
    let mut pidx = a.idx;
    loop {
        let p = &node!(ll, pidx);
        let ta = p.i != a.i;
        let tb = next!(ll, pidx).i != a.i;
        let tc = p.i != b.i;
        let td = next!(ll, pidx).i != b.i;
        let te = intersects(&p, &next!(ll, pidx), a, b);
        if ta && tb && tc && td && te {
            return true;
        }
        pidx = p.next_idx;
        if pidx == a.idx {
            break;
        }
    }
    return false;
}

// check if a polygon diagonal is locally inside the polygon
fn locally_inside(ll: &LL, a: &Node, b: &Node) -> bool {
    if area(&prev!(ll, a.idx), a, &next!(ll, a.idx)) < 0.0 {
        return area(a, b, &next!(ll, a.idx)) >= 0.0 && area(a, &prev!(ll, a.idx), b) >= 0.0;
    } else {
        return area(a, b, &prev!(ll, a.idx)) < 0.0 || area(a, &next!(ll, a.idx), b) < 0.0;
    }
}

// check if the middle point of a polygon diagonal is inside the polygon
fn middle_inside(ll: &LL, a: &Node, b: &Node) -> bool {
    let mut pi = a.idx;
    let mut inside = false;
    let px = (a.x + b.x) / 2.0;
    let py = (a.y + b.y) / 2.0;
    loop {
        let p = &node!(ll, pi);
        let pnext = &next!(ll, pi);

        if ((p.y > py) != (pnext.y > py))
            && (pnext.y != p.y)
            && (px < ((pnext.x - p.x) * (py - p.y) / (pnext.y - p.y) + p.x))
        {
            inside = !inside;
        }
        pi = next!(ll, pi).idx;
        if pi == a.idx {
            break;
        }
    }

    return inside;
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
same x y coordinates, as do 7 and 4.

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
fn split_polygon(ll: &mut LL, a: NodeIdx, b: NodeIdx) -> NodeIdx {
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
fn deviation(data: &Vec<f32>, hole_indices: &Vec<usize>, dim: usize, triangles: Vec<usize>) -> f32 {
    let has_holes = hole_indices.len() > 0;
    let mut outer_len = data.len();
    if has_holes {
        outer_len = hole_indices[0] * dim;
    }

    let mut polygon_area = signed_area(&data, 0, outer_len, dim).abs();
    if has_holes {
        let mut i = 0;
        let length = hole_indices.len();
        while i < length {
            i += 1;
            let start = hole_indices[i] * dim;
            let mut end = data.len();
            if i < length - 1 {
                end = hole_indices[i + 1] * dim;
            }
            polygon_area -= signed_area(&data, start, end, dim).abs();
        }
    }

    let mut triangles_area = 0.0f32;
    for i in (0..triangles.len()).step_by(3) {
        let a = triangles[i] * dim;
        let b = triangles[i + 1] * dim;
        let c = triangles[i + 2] * dim;
        triangles_area += ((data[a] - data[c]) * (data[b + 1] - data[a + 1])
            - (data[a] - data[b]) * (data[c + 1] - data[a + 1]))
            .abs();
    }

    if polygon_area == 0.0 && triangles_area == 0.0 {
        return 0.0;
    } else {
        return (triangles_area - polygon_area) / polygon_area;
    }
}

fn signed_area(data: &Vec<f32>, start: usize, end: usize, dim: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut j = end - dim;
    for i in (start..end).step_by(dim) {
        sum += (data[j] - data[i]) * (data[i + 1] + data[j + 1]);
        j = i;
    }
    return sum;
}

pub fn flatten(data: &Vec<Vec<Vec<f32>>>) -> (Vec<f32>, Vec<usize>, usize) {
    let mut coordinates: Vec<f32> = Vec::new();
    let mut hole_indexes: Vec<usize> = Vec::new();
    let dimensions = data[0][0].len();
    for i in 0..data.len() {
        for j in 0..data[i].len() {
            for d in 0..data[i][j].len() {
                coordinates.push(data[i][j][d]);
                //print!("{},",data[i][j][d]);
            }
        }
        if i > 0 {
            hole_indexes.push(data[i - 1].len());
        }
    }
    return (coordinates, hole_indexes, dimensions);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_llist() {
        let dims = 2;
        let data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let mut ll = linked_list(&data, 0, data.len(), dims, true);
        assert!(
            ll.nodes.len() == 4
                && ll.nodes[0].idx == 0
                && ll.nodes[0].i == 6
                && ll.nodes[0].x == 1.0
        );
        assert!(ll.nodes[0].i == 6 && ll.nodes[0].y == 0.0);
        assert!(ll.nodes[0].next_idx == 1 && ll.nodes[0].prev_idx == 3);
        assert!(ll.nodes[3].next_idx == 0 && ll.nodes[3].prev_idx == 2);
        println!("{}", ll.dump());
        ll.remove_node(2);
        println!("removed 2\n{}", ll.dump());
    }

    #[test]
    fn test_point_in_triangle() {
        assert!(point_in_triangle(0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.1));
        assert!(!point_in_triangle(0.0, 0.0, 2.0, 0.0, 2.0, 2.0, -1.0, 0.1));
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
        assert!(deviation(&data1, &hi, 2, tris) == 0.0);
    }

    #[test]
    fn test_get_leftmost() {
        let data = vec![-1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let ll = linked_list(&data, 0, data.len(), 2, true);
        assert!(node!(ll, get_leftmost(&ll, 2)).x == -1.0);
    }

    #[test]
    fn test_split_polygon() {
        let mut body = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let hole = vec![0.1, 0.1, 0.1, 0.2, 0.2, 0.2];
        body.extend(hole);
        let dims = 2;
        let mut ll = linked_list(&body, 0, body.len(), dims, true);
        assert!(ll.cycle_len(0) == body.len() / dims);
        let (left, right) = (0, 4);
        let np = split_polygon(&mut ll, left, right);
        assert!(ll.cycle_len(left) == 4);
        assert!(ll.cycle_len(np) == 5);
        // contrary to name, this should join the two cycles back together.
        let np2 = split_polygon(&mut ll, left, np);
        assert!(ll.cycle_len(np2) == 11);
        assert!(ll.cycle_len(left) == 11);
    }

    #[test]
    fn test_equals() {
        let dims = 2;

        let body = vec![0.0, 1.0, 0.0, 1.0];
        let ll = linked_list(&body, 0, body.len(), dims, true);
        assert!(equals(&ll.nodes[0], &ll.nodes[1]));

        let body = vec![2.0, 1.0, 0.0, 1.0];
        let ll = linked_list(&body, 0, body.len(), dims, true);
        assert!(!equals(&ll.nodes[0], &ll.nodes[1]));
    }

    #[test]
    fn test_area() {
        let dims = 2;
        let body = vec![4.0, 0.0, 4.0, 3.0, 0.0, 0.0]; // counterclockwise
        let ll = linked_list(&body, 0, body.len(), dims, true);
        assert!(area(&ll.nodes[0], &ll.nodes[1], &ll.nodes[2]) == -12.0);
        let body2 = vec![4.0, 0.0, 0.0, 0.0, 4.0, 3.0]; // clockwise
        let ll2 = linked_list(&body2, 0, body2.len(), dims, true);
        // creation apparently modifies all winding to ccw
        println!("{}", ll.dump());
        assert!(area(&ll.nodes[0], &ll.nodes[1], &ll.nodes[2]) == -12.0);
    }

    #[test]
    fn test_is_ear() {
        let dims = 2;
        let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0];
        let ll = linked_list(&m, 0, m.len(), dims, true);
        assert!(!is_ear(&ll, 0));
        assert!(!is_ear(&ll, 1));
        assert!(!is_ear(&ll, 2));

        let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.4];
        let ll = linked_list(&m, 0, m.len(), dims, true);
        assert!(is_ear(&ll, 0) == false);
        assert!(is_ear(&ll, 1) == true);
        assert!(is_ear(&ll, 2) == false);
        assert!(is_ear(&ll, 3) == true);

        let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0];
        let ll = linked_list(&m, 0, m.len(), dims, true);
        assert!(is_ear(&ll, 1));

        let m = vec![0.0, 0.0, 4.0, 0.0, 4.0, 3.0];
        let ll = linked_list(&m, 0, m.len(), dims, true);
        assert!(is_ear(&ll, 1));
    }

    #[test]
    fn test_filter_points() {
        let dims = 2;

        let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let mut ll1 = linked_list(&m, 0, m.len(), dims, true);
        let ll1len = ll1.nodes.len();
        let r1 = filter_points(&mut ll1, 0, ll1len - 1);
        assert!(ll1.cycle_len(r1) == 4);

        let n = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let mut ll2 = linked_list(&n, 0, n.len(), dims, true);
        let ll2len = ll2.nodes.len();
        let r2 = filter_points(&mut ll2, 0, ll2len - 1);
        assert!(ll2.cycle_len(r2) == 4);

        let n2 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let mut ll22 = linked_list(&n2, 0, n2.len(), dims, true);
        let r32 = filter_points(&mut ll22, 0, 99);
        assert!(ll22.cycle_len(r32) != 4);

        let o = vec![0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let mut ll3 = linked_list(&o, 0, o.len(), dims, true);
        let ll3len = ll3.nodes.len();
        let r3 = filter_points(&mut ll3, 0, ll3len - 1);
        assert!(ll3.cycle_len(r3) == 5);
    }

    #[test]
    fn test_earcut_linked() {
        let dim = 2;

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let mut ll = linked_list(&m, 0, m.len(), dim, true);
        let (mut tris, minx, miny, invsize, pass) = (Vec::new(), 0.0, 0.0, 0.0, 0);
        earcut_linked(&mut ll, 0, &mut tris, dim, minx, miny, invsize, 0);
        assert!(tris.len() == 6);

        let m = vec![0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let mut ll = linked_list(&m, 0, m.len(), dim, true);
        let (mut tris, minx, miny, invsize, pass) = (Vec::new(), 0.0, 0.0, 0.0, 0);
        earcut_linked(&mut ll, 0, &mut tris, dim, minx, miny, invsize, 0);
        assert!(tris.len() == 9);

        let m = vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let mut ll = linked_list(&m, 0, m.len(), dim, true);
        let (mut tris, minx, miny, invsize, pass) = (Vec::new(), 0.0, 0.0, 0.0, 0);
        earcut_linked(&mut ll, 0, &mut tris, dim, minx, miny, invsize, 0);
        assert!(tris.len() == 9);
    }

    #[test]
    fn test_middle_inside() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let ll = linked_list(&m, 0, m.len(), dim, true);
        assert!(middle_inside(&ll, &node!(ll, 0), &node!(ll, 2)));
        assert!(middle_inside(&ll, &node!(ll, 1), &node!(ll, 3)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let ll = linked_list(&m, 0, m.len(), dim, true);
        assert!(!middle_inside(&ll, &node!(ll, 0), &node!(ll, 2)));
        assert!(middle_inside(&ll, &node!(ll, 1), &node!(ll, 3)));
    }

    #[test]
    fn test_locally_inside() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let ll = linked_list(&m, 0, m.len(), dim, true);
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 0)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 1)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 2)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 3)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let ll = linked_list(&m, 0, m.len(), dim, true);
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 0)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 1)));
        assert!(!locally_inside(&ll, &node!(ll, 0), &node!(ll, 2)));
        assert!(locally_inside(&ll, &node!(ll, 0), &node!(ll, 3)));
    }

    #[test]
    fn test_intersects_polygon() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let ll = linked_list(&m, 0, m.len(), dim, true);
        /*
		println!("{}",intersects_polygon( &ll, &node!(ll,0),&node!(ll,2)) );
		println!("{}",intersects_polygon( &ll, &node!(ll,2),&node!(ll,0)) );
		println!("{}",intersects_polygon( &ll, &node!(ll,1),&node!(ll,3)) );
		println!("{}",intersects_polygon( &ll, &node!(ll,3),&node!(ll,1)) );
*/
        println!("-1");

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 1.0, 0.0, 1.0];
        let ll = linked_list(&m, 0, m.len(), dim, true);
        println!("{}", intersects_polygon(&ll, &node!(ll, 0), &node!(ll, 2)));
        println!("-2");
        println!("{}", intersects_polygon(&ll, &node!(ll, 2), &node!(ll, 0)));
        println!("{}", ll.dump());
        /*		println!("{}",intersects_polygon( &ll, &node!(ll,1),&node!(ll,3)) );
		println!("{}",intersects_polygon( &ll, &node!(ll,3),&node!(ll,1)) );
		println!("{}",intersects_polygon( &ll, &node!(ll,1),&node!(ll,5)) );
		println!("{}",intersects_polygon( &ll, &node!(ll,5),&node!(ll,1)) );
		println!("{}",intersects_polygon( &ll, &node!(ll,1),&node!(ll,4)) );
		println!("{}",intersects_polygon( &ll, &node!(ll,4),&node!(ll,1)) );
*/
    }

    #[test]
    fn test_intersects() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 0.9, 0.9, 0.0, 1.0];
        let ll = linked_list(&m, 0, m.len(), dim, true);
		macro_rules! ti {
		    ($ok:expr,$a:expr,$b:expr,$c:expr,$d:expr) => {
				assert!($ok==intersects(&ll.nodes[$a],&ll.nodes[$b],&ll.nodes[$c],&ll.nodes[$d]));
		    };
		};
		ti!(true,0,2,0,1);
		ti!(true,0,2,1,2);
		ti!(true,0,2,2,3);
		ti!(true,0,2,3,0);
		ti!(true,0,2,3,1);
		ti!(true,0,2,1,3);
		ti!(true,0,2,2,0);
		ti!(true,0,2,0,2);
		ti!(false,0,1,2,3);
		ti!(false,1,0,2,3);
		ti!(false,0,0,2,3);
		ti!(false,0,1,3,2);
		ti!(false,1,0,3,2);
    }

	#[test]
	fn test_is_valid_diagonal() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let ll = linked_list(&m, 0, m.len(), dim, true);
		assert!(!is_valid_diagonal(&ll,&ll.nodes[0],&ll.nodes[1]));
		assert!(!is_valid_diagonal(&ll,&ll.nodes[1],&ll.nodes[2]));
		assert!(!is_valid_diagonal(&ll,&ll.nodes[2],&ll.nodes[3]));
		assert!(!is_valid_diagonal(&ll,&ll.nodes[3],&ll.nodes[0]));
		assert!(!is_valid_diagonal(&ll,&ll.nodes[0],&ll.nodes[2]));
		assert!(is_valid_diagonal(&ll,&ll.nodes[1],&ll.nodes[3]));
		assert!(!is_valid_diagonal(&ll,&ll.nodes[2],&ll.nodes[0]));
		assert!(is_valid_diagonal(&ll,&ll.nodes[3],&ll.nodes[1]));
	}

    #[test]
    fn test_bbox_intersect() {
        let dim = 2;
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let ll = linked_list(&m, 0, m.len(), dim, true);
		println!("{}",ll.dump());
        println!(
            "{}",
            bbox_intersect(&ll.nodes[0], &ll.nodes[1], &ll.nodes[2], &ll.nodes[3])
        );

        let m = vec![-1.0, -1.0, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0];
        let ll = linked_list(&m, 0, m.len(), dim, true);
		println!("{}",ll.dump());
        assert!(!bbox_intersect(&ll.nodes[0], &ll.nodes[1], &ll.nodes[2], &ll.nodes[3]));
        assert!(!bbox_intersect(&ll.nodes[0], &ll.nodes[1], &ll.nodes[3], &ll.nodes[2]));
 assert!(bbox_intersect(&ll.nodes[0], &ll.nodes[2], &ll.nodes[1], &ll.nodes[3]));
 assert!(bbox_intersect(&ll.nodes[0], &ll.nodes[2], &ll.nodes[3], &ll.nodes[1]));
 assert!(bbox_intersect(&ll.nodes[0], &ll.nodes[3], &ll.nodes[3], &ll.nodes[3]));
 assert!(bbox_intersect(&ll.nodes[0], &ll.nodes[3], &ll.nodes[1], &ll.nodes[2]));
 assert!(bbox_intersect(&ll.nodes[2], &ll.nodes[0], &ll.nodes[1], &ll.nodes[3]));
 assert!(bbox_intersect(&ll.nodes[2], &ll.nodes[0], &ll.nodes[3], &ll.nodes[1]));
 assert!(bbox_intersect(&ll.nodes[3], &ll.nodes[0], &ll.nodes[3], &ll.nodes[3]));
 assert!(bbox_intersect(&ll.nodes[3], &ll.nodes[0], &ll.nodes[1], &ll.nodes[2]));
    }

}
