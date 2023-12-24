static DIM: usize = 2;
static NULL: usize = 0;
//static DEBUG: usize = 4;
static DEBUG: usize = 0; // dlogs get optimized away at 0

type NodeIdx = usize;
type VertIdx = usize;

mod node;
use node::*;

fn compare_x(a: &Node, b: &Node) -> std::cmp::Ordering {
    a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
}

// add new nodes to an existing linked list.
pub fn linked_list_add_contour(
    ll: &mut LinkedLists,
    data: &Vec<f64>,
    start: usize,
    end: usize,
    clockwise: bool,
) -> (NodeIdx, NodeIdx) {
    if start > data.len() || end > data.len() || data.len() == 0 {
        return (NULL, NULL);
    }
    let mut lastidx = NULL;
    let mut leftmost_idx = NULL;
    let mut contour_minx = std::f64::MAX;

    if clockwise == (signed_area(&data, start, end) > 0.0) {
        for i in (start..end).step_by(DIM) {
            lastidx = ll.insert_node(i / DIM, data[i], data[i + 1], lastidx);
            if contour_minx > data[i] {
                contour_minx = data[i];
                leftmost_idx = lastidx
            };
            ll.miny = f64::min(data[i + 1], ll.miny);
            ll.maxx = f64::max(data[i], ll.maxx);
            ll.maxy = f64::max(data[i + 1], ll.maxy);
        }
    } else {
        for i in (start..=(end - DIM)).rev().step_by(DIM) {
            lastidx = ll.insert_node(i / DIM, data[i], data[i + 1], lastidx);
            if contour_minx > data[i] {
                contour_minx = data[i];
                leftmost_idx = lastidx
            };
            ll.miny = f64::min(data[i + 1], ll.miny);
            ll.maxx = f64::max(data[i], ll.maxx);
            ll.maxy = f64::max(data[i + 1], ll.maxy);
        }
    }

    ll.minx = f64::min(contour_minx, ll.minx);

    if equal_coords(lastidx.node(ll), lastidx.next(ll).node(ll)) {
        ll.remove_node(lastidx);
        lastidx = lastidx.next(ll)
    }
    return (lastidx, leftmost_idx);
}

// link every hole into the outer loop, producing a single-ring polygon
// without holes
fn eliminate_holes(
    ll: &mut LinkedLists,
    data: &Vec<f64>,
    hole_indices: &Vec<usize>,
    inouter_node: NodeIdx,
) -> NodeIdx {
    let mut queue: Vec<Node> = Vec::new();
    for i in 0..hole_indices.len() {
        let start = hole_indices[i] * DIM;
        let end = if i < (hole_indices.len() - 1) {
            hole_indices[i + 1] * DIM
        } else {
            data.len()
        };
        let (list, leftmost_idx) = linked_list_add_contour(ll, &data, start, end, false);
        if list == list.next(ll) {
            list.set_steiner(ll, true);
        }
        queue.push(leftmost_idx.node(ll).clone());
    }

    queue.sort_by(compare_x);

    queue.iter().fold(inouter_node, |outer_node, qnode| {
        eliminate_hole(ll, qnode.idx, outer_node);
        filter_points(ll, outer_node, outer_node.next(ll))
    })
} // elim holes

// minx, miny and invsize are later used to transform coords
// into integers for z-order calculation
fn calc_invsize(minx: f64, miny: f64, maxx: f64, maxy: f64) -> f64 {
    let invsize = f64::max(maxx - minx, maxy - miny);
    match invsize == 0.0 {
        true => 0.0,
        false => 32767.0 / invsize,
    }
}

// main ear slicing loop which triangulates a polygon (given as a linked
// list)
fn earcut_linked_hashed(
    ll: &mut LinkedLists,
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
    let mut next_idx = ear_idx.next(ll);
    while stop_idx != next_idx {
        prev_idx = ear_idx.prev(ll);
        next_idx = ear_idx.next(ll);
        if is_ear_hashed(ll, prev_idx, ear_idx, next_idx) {
            triangles.push(prev_idx.node(ll).i);
            triangles.push(ear_idx.node(ll).i);
            triangles.push(next_idx.node(ll).i);
            ll.remove_node(ear_idx);
            // skipping the next vertex leads to less sliver triangles
            ear_idx = next_idx.node(ll).next_idx;
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

// interlink polygon nodes in z-order
fn index_curve(ll: &mut LinkedLists, start: NodeIdx) {
    let invsize = ll.invsize;
    let mut p = start;
    loop {
        p.set_z(ll, zorder(p.node(ll).x, p.node(ll).y, invsize));
        p.set_prevz(ll, p.prev(ll));
        p.set_nextz(ll, p.next(ll));
        p = p.next(ll);
        if p == start {
            break;
        }
    }

    start.prevz(ll).set_nextz(ll, NULL);
    start.set_prevz(ll, NULL);
    sort_linked(ll, start);
}

// Simon Tatham's linked list merge sort algorithm
// http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
fn sort_linked(ll: &mut LinkedLists, mut list: NodeIdx) {
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
                q = q.nextz(ll);
            }
            qsize = insize;

            while psize > 0 || (qsize > 0 && q != NULL) {
                if psize > 0 && (qsize == 0 || q == NULL || p.node(ll).z <= q.node(ll).z) {
                    e = p;
                    p = p.nextz(ll);
                    psize -= 1;
                } else {
                    e = q;
                    q = q.nextz(ll);
                    qsize -= 1;
                }

                if tail != NULL {
                    tail.set_nextz(ll, e);
                } else {
                    list = e;
                }

                e.set_prevz(ll, tail);
                tail = e;
            }

            p = q;
        }

        tail.set_nextz(ll, NULL);
        insize *= 2;
        if nummerges <= 1 {
            break;
        }
    }
}

// check whether a polygon node forms a valid ear with adjacent nodes
fn is_ear(ll: &LinkedLists, prev: NodeIdx, ear: NodeIdx, next: NodeIdx) -> bool {
    let (a, b, c) = (prev.node(ll), ear.node(ll), next.node(ll));
    match area(a, b, c) >= 0.0 {
        true => false, // reflex, cant be ear
        false => !ll.iter(c.next_idx..a.idx).any(|p| {
            point_in_triangle(&a, &b, &c, &p)
                && (area(p.idx.prev(ll).node(ll), &p, p.idx.next(ll).node(ll)) >= 0.0)
        }),
    }
}

// helper for is_ear_hashed. needs manual inline (rust 2018)
#[inline(always)]
fn earcheck(a: &Node, b: &Node, c: &Node, prev: &Node, p: &Node, next: &Node) -> bool {
    (p.idx != a.idx)
        && (p.idx != c.idx)
        && point_in_triangle(&a, &b, &c, &p)
        && area(&prev, &p, &next) >= 0.0
}

#[inline(always)]
fn is_ear_hashed(
    ll: &mut LinkedLists,
    prev_idx: NodeIdx,
    ear_idx: NodeIdx,
    next_idx: NodeIdx,
) -> bool {
    let (prev, ear, next) = (prev_idx.node(ll), ear_idx.node(ll), next_idx.node(ll));
    if area(&prev, &ear, &next) >= 0.0 {
        return false;
    };

    let bbox_maxx = f64::max(prev.x, f64::max(ear.x, next.x));
    let bbox_maxy = f64::max(prev.y, f64::max(ear.y, next.y));
    let bbox_minx = f64::min(prev.x, f64::min(ear.x, next.x));
    let bbox_miny = f64::min(prev.y, f64::min(ear.y, next.y));
    // z-order range for the current triangle bbox;
    let min_z = zorder(bbox_minx, bbox_miny, ll.invsize);
    let max_z = zorder(bbox_maxx, bbox_maxy, ll.invsize);

    macro_rules! earcheck_or_exit {
        ($p:expr,$e:expr,$n:expr,$o:expr) => {
            if earcheck(
                &$p,
                &$e,
                &$n,
                $o.prev(ll).node(ll),
                $o.node(ll),
                $o.next(ll).node(ll),
            ) {
                return false;
            }
        };
    }

    let mut p = ear.prevz_idx;
    let mut n = ear.nextz_idx;
    while p != NULL && p.node(ll).z >= min_z && n != NULL && n.node(ll).z <= max_z {
        earcheck_or_exit!(&prev, &ear, &next, p);
        p = p.prevz(ll);

        earcheck_or_exit!(&prev, &ear, &next, n);
        n = n.nextz(ll);
    }

    while p.node(ll).z >= min_z && p != NULL {
        earcheck_or_exit!(&prev, &ear, &next, p);

        p = p.prevz(ll);
    }

    while n.node(ll).z <= max_z && n != NULL {
        earcheck_or_exit!(&prev, &ear, &next, n);

        n = n.nextz(ll);
    }

    true
}

fn filter_points(ll: &mut LinkedLists, start: NodeIdx, mut end: NodeIdx) -> NodeIdx {
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

    // this loop "wastes" calculations by going over the same points multiple
    // times. however, altering the location of the 'end' node can disrupt
    // the algorithm of other code that calls the filter_points function.
    loop {
        again = false;
        if !p.node(ll).steiner
            && (equal_coords(p.node(ll), p.next(ll).node(ll))
                || area(p.prev(ll).node(ll), p.node(ll), p.next(ll).node(ll)) == 0.0)
        {
            ll.remove_node(p);
            end = p.prev(ll);
            p = end;
            if p == p.next(ll) {
                break end;
            }
            again = true;
        } else {
            p = p.next(ll);
        }
        if !again && p == end {
            break end;
        }
    }
}

// create a circular doubly linked list from polygon points in the
// specified winding order
fn linked_list(
    data: &Vec<f64>,
    start: usize,
    end: usize,
    clockwise: bool,
) -> (LinkedLists, NodeIdx) {
    let mut ll: LinkedLists = LinkedLists::new(data.len() / DIM);
    let (last_idx, _) = linked_list_add_contour(&mut ll, data, start, end, clockwise);
    (ll, last_idx)
}

// z-order of a point given coords and inverse of the longer side of
// data bbox
#[inline(always)]
fn zorder(xf: f64, yf: f64, invsize: f64) -> i32 {
    // coords are transformed into non-negative 15-bit integer range
    // stored in two 32bit ints, which are combined into a single 64 bit int.
    let x: i64 = (xf * invsize) as i64;
    let y: i64 = (yf * invsize) as i64;
    let mut xy: i64 = x << 32 | y;

    // todo ... big endian?
    xy = (xy | (xy << 8)) & 0x00FF00FF00FF00FF;
    xy = (xy | (xy << 4)) & 0x0F0F0F0F0F0F0F0F;
    xy = (xy | (xy << 2)) & 0x3333333333333333;
    xy = (xy | (xy << 1)) & 0x5555555555555555;

    ((xy >> 32) | (xy << 1)) as i32
}

fn wedge(a: Node, b: Node) -> f64 {
    a.x * b.y - b.x * a.y
}

// check if a point lies within a convex triangle
fn point_in_triangle(a: &Node, b: &Node, c: &Node, p: &Node) -> bool {
    wedge(c - p, a - p) >= 0.0 && wedge(a - p, b - p) >= 0.0 && wedge(b - p, c - p) >= 0.0
}

pub fn earcut(data: &Vec<f64>, hole_indices: &Vec<usize>, dims: usize) -> Vec<usize> {
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

    ll.invsize = calc_invsize(ll.minx, ll.miny, ll.maxx, ll.maxy);

    // translate all points so min is 0,0. prevents subtraction inside
    // zorder. also note invsize does not depend on translation in space
    // if one were translating in a space with an even spaced grid of points.
    // floating point space is not evenly spaced, but it is close enough for
    // this hash algorithm
    let (mx, my) = (ll.minx, ll.miny);
    ll.nodes.iter_mut().for_each(|n| n.x -= mx);
    ll.nodes.iter_mut().for_each(|n| n.y -= my);
    earcut_linked_hashed(&mut ll, outer_node, &mut triangles, 0);

    triangles
}

// signed area of a parallelogram
fn area(p: &Node, q: &Node, r: &Node) -> f64 {
    (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
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
        let a = p.prev(ll);
        let b = p.next(ll).next(ll);

        if !equal_coords( a.node(ll), b.node(ll))
            && pseudo_intersects(
                a.node(ll),
                p.node(ll),
                p.next(ll).node(ll),
                b.node(ll),
            )
			// prev next a, prev next b
            && locally_inside(ll, a.node(ll), b.node(ll))
            && locally_inside(ll, b.node(ll), a.node(ll))
        {
            triangles.push(a.node(ll).i);
            triangles.push(p.node(ll).i);
            triangles.push(b.node(ll).i);

            // remove two nodes involved
            ll.remove_node(p);
            let nidx = p.node(ll).next_idx;
            ll.remove_node(nidx);

            start = b.node(ll).idx;
            p = start;
        }
        p = p.node(ll).next_idx;
        if p == start {
            break;
        }
    }

    return p;
}

// try splitting polygon into two and triangulate them independently
fn split_earcut(ll: &mut LinkedLists, start_idx: NodeIdx, triangles: &mut Vec<NodeIdx>) {
    // look for a valid diagonal that divides the polygon into two
    let mut a = start_idx;
    loop {
        let mut b = a.next(ll).next(ll);
        while b != a.node(ll).prev_idx {
            if a.node(ll).i != b.node(ll).i && is_valid_diagonal(ll, a.node(ll), b.node(ll)) {
                // split the polygon in two by the diagonal
                let mut c = split_bridge_polygon(ll, a, b);

                // filter colinear points around the cuts
                let an = a.node(ll).next_idx;
                let cn = c.node(ll).next_idx;
                a = filter_points(ll, a, an);
                c = filter_points(ll, c, cn);

                // run earcut on each half
                earcut_linked_hashed(ll, a, triangles, 0);
                earcut_linked_hashed(ll, c, triangles, 0);
                return;
            }
            b = b.node(ll).next_idx;
        }
        a = a.node(ll).next_idx;
        if a == start_idx {
            break;
        }
    }
}

// find a bridge between vertices that connects hole with an outer ring
// and and link it
fn eliminate_hole(ll: &mut LinkedLists, hole_idx: NodeIdx, outer_node_idx: NodeIdx) {
    let test_idx = find_hole_bridge(ll, hole_idx, outer_node_idx);
    let b = split_bridge_polygon(ll, test_idx, hole_idx);
    let ni = b.next(ll);
    filter_points(ll, b, ni);
}

// David Eberly's algorithm for finding a bridge between hole and outer polygon
fn find_hole_bridge(ll: &LinkedLists, hole: NodeIdx, outer_node: NodeIdx) -> NodeIdx {
    let mut p = outer_node;
    let hx = hole.node(ll).x;
    let hy = hole.node(ll).y;
    let mut qx: f64 = std::f64::NEG_INFINITY;
    let mut m: NodeIdx = NULL;

    // find a segment intersected by a ray from the hole's leftmost
    // point to the left; segment's endpoint with lesser x will be
    // potential connection point
    let calcx = |p: &Node| p.x + (hy - p.y) * (p.next(ll).x - p.x) / (p.next(ll).y - p.y);
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
        return m.prev(ll);
    }

    // look for points inside the triangle of hole point, segment
    // intersection and endpoint; if there are no points found, we have
    // a valid connection; otherwise choose the point of the minimum
    // angle with the ray as connection point

    let mp = Node::new(0, m.node(ll).x, m.node(ll).y, 0);
    p = m.next(ll);
    let x1 = if hy < mp.y { hx } else { qx };
    let x2 = if hy < mp.y { qx } else { hx };
    let n1 = Node::new(0, x1, hy, 0);
    let n2 = Node::new(0, x2, hy, 0);

    let calctan = |p: &Node| (hy - p.y).abs() / (hx - p.x); // tangential
    ll.iter(p..m)
        .filter(|p| hx > p.x && p.x >= mp.x)
        .filter(|p| point_in_triangle(&n1, &mp, &n2, &p))
        .fold((m, std::f64::MAX / 2.), |(m, tan_min), p| {
            if ((calctan(p) < tan_min) || (calctan(p) == tan_min && p.x > m.node(ll).x))
                && locally_inside(ll, &p, hole.node(ll))
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
fn is_valid_diagonal(ll: &LinkedLists, a: &Node, b: &Node) -> bool {
    return a.next(ll).i != b.i
        && a.prev(ll).i != b.i
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
    if (equal_coords(p1, p2) && equal_coords(q1, q2))
        || (equal_coords(p1, q2) && equal_coords(q1, p2))
    {
        return true;
    }
    return (area(p1, q1, p2) > 0.0) != (area(p1, q1, q2) > 0.0)
        && (area(p2, q2, p1) > 0.0) != (area(p2, q2, q1) > 0.0);
}

// check if a polygon diagonal intersects any polygon segments
fn intersects_polygon(ll: &LinkedLists, a: &Node, b: &Node) -> bool {
    ll.iter_pairs(a.idx..a.idx).any(|(p, n)| {
        p.i != a.i && n.i != a.i && p.i != b.i && n.i != b.i && pseudo_intersects(&p, &n, a, b)
    })
}

// check if a polygon diagonal is locally inside the polygon
fn locally_inside(ll: &LinkedLists, a: &Node, b: &Node) -> bool {
    match area(a.prev(ll), a, a.next(ll)) < 0.0 {
        true => area(a, b, a.next(ll)) >= 0.0 && area(a, a.prev(ll), b) >= 0.0,
        false => area(a, b, a.prev(ll)) < 0.0 || area(a, a.next(ll), b) < 0.0,
    }
}

// check if the middle point of a polygon diagonal is inside the polygon
fn middle_inside(ll: &LinkedLists, a: &Node, b: &Node) -> bool {
    let (mx, my) = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
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
fn split_bridge_polygon(ll: &mut LinkedLists, a: NodeIdx, b: NodeIdx) -> NodeIdx {
    let cidx = ll.nodes.len();
    let didx = cidx + 1;
    let mut c = Node::new(a.node(ll).i, a.node(ll).x, a.node(ll).y, cidx);
    let mut d = Node::new(b.node(ll).i, b.node(ll).x, b.node(ll).y, didx);

    let an = a.next(ll);
    let bp = b.prev(ll);

    a.set_next(ll, b);
    b.set_prev(ll, a);

    c.next_idx = an;
    an.set_prev(ll, cidx);

    d.next_idx = cidx;
    c.prev_idx = didx;

    bp.set_next(ll, didx);
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
    dims: usize,
    triangles: &Vec<usize>,
) -> f64 {
    if DIM != dims {
        return std::f64::NAN;
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

fn signed_area(data: &Vec<f64>, start: usize, end: usize) -> f64 {
    let i = (start..end).step_by(DIM);
    let j = (start..end).cycle().skip((end - DIM) - start).step_by(DIM);
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
            })
            .collect::<Vec<usize>>(), // hole indexes
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

fn cycle_dump(ll: &LinkedLists, p: NodeIdx) -> String {
    let mut s = format!("cycle from {}, ", p);
    s.push_str(&format!(" len {}, idxs:", 0)); //cycle_len(&ll, p)));
    let mut i = p;
    let end = i;
    let mut count = 0;
    loop {
        count += 1;
        s.push_str(&format!("{} ", i.node(ll).idx));
        s.push_str(&format!("(i:{}), ", i.node(ll).i));
        i = i.node(ll).next_idx;
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
    fn test_equal_coords() {
        let body = vec![0.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&body, 0, body.len(), true);
        assert!(equal_coords(&ll.nodes[1], &ll.nodes[2]));

        let body = vec![2.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&body, 0, body.len(), true);
        assert!(!equal_coords(&ll.nodes[1], &ll.nodes[2]));
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
        assert!(middle_inside(&ll, ll.node(1), ll.node(3)));
        assert!(middle_inside(&ll, ll.node(2), ll.node(4)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(!middle_inside(&ll, ll.node(1), ll.node(3)));
        assert!(middle_inside(&ll, ll.node(2), ll.node(4)));
    }

    #[test]
    fn test_locally_inside() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(locally_inside(&ll, ll.node(1), ll.node(1)));
        assert!(locally_inside(&ll, ll.node(1), ll.node(2)));
        assert!(locally_inside(&ll, ll.node(1), ll.node(3)));
        assert!(locally_inside(&ll, ll.node(1), ll.node(4)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 0.1];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        assert!(locally_inside(&ll, ll.node(1), ll.node(1)));
        assert!(locally_inside(&ll, ll.node(1), ll.node(2)));
        assert!(!locally_inside(&ll, ll.node(1), ll.node(3)));
        assert!(locally_inside(&ll, ll.node(1), ll.node(4)));
    }

    #[test]
    fn test_intersects_polygon() {
        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);

        assert!(false == intersects_polygon(&ll, ll.node(0), ll.node(2)));
        assert!(false == intersects_polygon(&ll, ll.node(2), ll.node(0)));
        assert!(false == intersects_polygon(&ll, ll.node(1), ll.node(3)));
        assert!(false == intersects_polygon(&ll, ll.node(3), ll.node(1)));

        let m = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&m, 0, m.len(), true);
        dlog!(9, "{}", dump(&ll));
        dlog!(5, "{}", intersects_polygon(&ll, ll.node(0), ll.node(2)));
        dlog!(5, "{}", intersects_polygon(&ll, ll.node(2), ll.node(0)));
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
        }
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
        assert!((4 as NodeIdx).next(&ll) == 1);
        assert!((1 as NodeIdx).next(&ll) == 2);
        ll.remove_node(2);
    }

    #[test]
    fn test_iter_pairs() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let (ll, _) = linked_list(&data, 0, data.len(), true);
        let mut v: Vec<Node> = Vec::new();
        //        ll.iter(1..2)
        //.zip(ll.iter(2..3))
        ll.iter_pairs(1..2).for_each(|(p, n)| {
            v.push(p.clone());
            v.push(n.clone());
        });
        println!("{:?}", v);
        //		assert!(false);
    }
}
