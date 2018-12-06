static NULL: usize = 0x777A91CC;
static DEBUG: bool = true;

impl Node {
    fn new(i: usize, x: f32, y: f32) -> Node {
        Node {
            i: i,
            x: x,
            y: y,
            prev: NULL,
            next: NULL,
            z: 0,
            nextz: NULL,
            prevz: NULL,
            steiner: false,
        }
    }
}

#[derive(Clone)]
struct Node {
    i: usize,    // vertex index in coordinates array
    x: f32,      // vertex x coordinates
    y: f32,      // vertex y coordinates
    prev: usize, // previous vertex nodes in a polygon ring
    next: usize,
    z: u32,       // z-order curve value
    prevz: usize, // previous and next nodes in z-order
    nextz: usize,
    steiner: bool, // indicates whether this is a steiner point
}

struct LL {
    nodes: Vec<Node>,
}

impl LL {
    fn dump(&self) -> String {
        fn pn(a: usize) -> String {
            if a == NULL {
                return String::from("NULL");
            } else {
                return a.to_string();
            }
        }
        let mut s = format!("LL, nodes: {}", self.nodes.len());
        s.push_str(&format!(
            " {:>3} {:>3} {:>3} {:>3} {:>6} {:>6} {:>3} {:>3} ",
            "vi", "i", "n", "p", "x", "y", "nz", "pz"
        ));
        for (vi, n) in self.nodes.iter().enumerate() {
            s.push_str(&format!(
                " {:>3} {:>3} {:>4} {:>4} {:>6} {:>6} {:>4} {:>4} ",
                vi,
                n.i,
                pn(n.next),
                pn(n.prev),
                n.x,
                n.y,
                pn(n.nextz),
                pn(n.prevz)
            ));
        }
        return s;
    }
    fn insert_node(&mut self, i: usize, x: f32, y: f32) {
        let ll = self;
        let ls = ll.nodes.len();
        let mut p = Node::new(i, x, y);
        if ls == 0 {
            p.next = 0;
            p.prev = 0;
        } else {
            p.next = ll.nodes[ls - 1].next;
            p.prev = ls - 1;
            let z = ll.nodes[ls - 1].next;
            ll.nodes[z].prev = ls;
            ll.nodes[ls - 1].next = ls;
        }
        ll.nodes.push(p.clone());
    }
    fn remove_node(&mut self, nodeidx: usize) {
        let ll = self;
        let p = &ll.nodes[nodeidx].clone();

        ll.nodes[p.next].prev = p.prev;
        ll.nodes[p.prev].next = p.next;
        //    p.next.prev = p.prev;
        //    p.prev.next = p.next;

        if p.prevz != NULL {
            ll.nodes[p.prevz].nextz = p.nextz;
        }
        if p.nextz != NULL {
            ll.nodes[p.nextz].prevz = p.prevz;
        }
        //    if (p.prevz) p.prevz.nextz = p.nextz;
        //    if (p.nextz) p.nextz.prevz = p.prevz;
        ll.nodes[nodeidx].next = NULL;
        ll.nodes[nodeidx].prev = NULL;
        ll.nodes[nodeidx].nextz = NULL;
        ll.nodes[nodeidx].prevz = NULL;
    }
    fn new() -> LL {
        LL { nodes: Vec::new() }
    }
    // link every hole into the outer loop, producing a single-ring polygon
    // without holes
    fn eliminate_holes(&mut self, data: &Vec<f32>, hole_indices: Vec<usize>, dim: usize) {
        /*    let queue = [],
    let (i, len, start, end, list) 

    for (i = 0, len = hole_indices.len(); i < len; i+=1) {
        start = hole_indices[i] * dim;
        end = i < len - 1 ? hole_indices[i + 1] * dim : data.len();
        list = linkedList(data, start, end, dim:usize, false);
        if (list == list.next) list.steiner = true;
        queue.push(getLeftmost(list));
    }

    queue.sort(compareX);

    // process holes from left to right
    for (i = 0; i < queue.len(); i+=1) {
        eliminateHole(queue[i], outerNode);
        outerNode = filter_points(outerNode, outerNode.next);
    }

    return outerNode;
*/
    } // elim holes

    // main ear slicing loop which triangulates a polygon (given as a linked
    // list)
    fn earcut_linked(
        &mut self,
        triangles: &mut Vec<usize>,
        dim: usize,
        minx: f32,
        miny: f32,
        invsize: f32,
        pass: usize,
    ) {
        if DEBUG {
            println!(
                "earcut_linked nodes:{} tris:{} dm:{} mx:{} my:{} invs:{} pas:{}",
                self.nodes.len(),
                triangles.len(),
                dim,
                minx,
                miny,
                invsize,
                pass
            );
        }
        if self.nodes.len() == 0 {
            return;
        }

        // interlink polygon nodes in z-order
        // note this does nothing for smaller data len, b/c invsize will be 0
        if pass == 0 && invsize > 0.0 {
            self.index_curve(0, minx, miny, invsize);
        }

        let mut stopidx = 0;
        let mut earidx = 0;
        let mut previdx = 0;
        let mut nextidx = 0;
        // iterate through ears, slicing them one by one
        while self.nodes[earidx].prev != self.nodes[earidx].next {
            if DEBUG {
                println!("p{} e{} n{} s{}", previdx, earidx, nextidx, stopidx);
            }
            if DEBUG {
                self.dump();
            }
            previdx = self.nodes[earidx].prev;
            nextidx = self.nodes[earidx].next;
            //            prev = v[ear].prev;
            //            next = v[ear].next;

            let mut test;
            if invsize > 0.0 {
                test = self.is_ear_hashed(earidx, minx, miny, invsize);
            } else {
                test = self.is_ear(earidx);
            }
            if test {
                // cut off the triangle
                triangles.push(self.nodes[previdx].i / dim);
                triangles.push(self.nodes[earidx].i / dim);
                triangles.push(self.nodes[nextidx].i / dim);

                self.remove_node(earidx);

                // skipping the next vertex leads to less sliver triangles
                earidx = self.nodes[nextidx].next;
                stopidx = self.nodes[nextidx].next;
                continue;
            }

            earidx = nextidx;

            // f we looped through the whole remaining polygon and can't
            // find any more ears
            if earidx == stopidx {
                /*
                        // try filtering points and slicing again
//            if (!pass) { earcut_linked(filter_points(earidx), triangles, 
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
        if DEBUG {
            println!("earcut_linked end");
        }
    } //cut_linked

    // interlink polygon nodes in z-order
    fn index_curve(&mut self, start: usize, minx: f32, miny: f32, invsize: f32) {
        if DEBUG {
            println!("index curve");
        }
        if DEBUG {
            self.dump();
        }
        let mut nodeidx = start;
        loop {
            let p = &mut self.nodes[nodeidx];
            if p.z == NULL as u32 {
                p.z = zorder(p.x, p.y, minx, miny, invsize);
            }
            p.prevz = p.prev;
            p.nextz = p.next;
            nodeidx = p.next;
            if nodeidx == start {
                break;
            }
        }

        let pzidx = self.nodes[nodeidx].prevz;
        self.nodes[pzidx].nextz = NULL;
        self.nodes[nodeidx].prevz = NULL;
        //    p.prevz.nextz = NULL;
        //    p.prevz = NULL;

        self.sort_linked();
        if DEBUG {
            self.dump();
        }
        if DEBUG {
            println!("index curve end");
        }
    } // indexcurve

    // Simon Tatham's linked list merge sort algorithm
    // http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
    fn sort_linked(&mut self) {
        if DEBUG {
            println!("sort linked");
        }
        if DEBUG {
            self.dump();
        }
        let (mut i, mut pidx, mut qidx, mut eidx, tailidx, mut nummerges, mut psize, mut qsize) =
            (0, 0, 0, 0, 0, 0, 0, 0);
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
                    qidx = self.nodes[qidx.clone()].nextz;
                    if qidx == NULL {
                        break;
                    }
                }
                qsize = insize;

                while psize > 0 || (qsize > 0 && qidx != NULL) {
                    if psize != 0
                        && (qsize == 0 || qidx == NULL || self.nodes[pidx].z <= self.nodes[qidx].z)
                    {
                        eidx = pidx;
                        pidx = self.nodes[pidx].nextz;
                        psize -= 1;
                    } else {
                        eidx = qidx;
                        qidx = self.nodes[qidx].nextz;
                        qsize -= 1;
                    }

                    if tailidx != NULL {
                        self.nodes[tailidx].nextz = eidx;
                    } else {
                        listidx = eidx;
                    }

                    self.nodes[eidx].prevz = tailidx;
                    tailidx = eidx;
                }

                pidx = qidx;
            }

            self.nodes[tailidx].nextz = NULL;
            insize *= 2;
            if nummerges <= 1 {
                break;
            }
        } // while (nummerges > 1);
        if DEBUG {
            self.dump();
        }
        if DEBUG {
            println!("sort linked end");
        }
    } // end sort

    // check whether a polygon node forms a valid ear with adjacent nodes
    fn is_ear(&mut self, earidx: usize) -> bool {
        if DEBUG {
            println!("is ear, {}", earidx as u32);
        }
        let result = true;
        let a = &self.nodes[self.nodes[earidx].prev];
        let b = &self.nodes[earidx];
        let c = &self.nodes[self.nodes[earidx].next];

        // reflex, can't be an ear
        if area(a, b, c) >= 0.0 {
            if DEBUG {
                println!("is ear false (Area>0)");
            }
            return false;
        }

        // now make sure we don't have other points inside the potential ear
        let mut pidx = self.nodes[b.next].next;

        while pidx != self.nodes[earidx].prev {
            let p = &self.nodes[pidx];
            let ppnode = &self.nodes[p.prev];
            let pnnode = &self.nodes[p.next];
            if point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y)
                && area(ppnode, p, pnnode) >= 0.0
            {
                if DEBUG {
                    println!("is ear false (pt in tri,Area>0)");
                }
                return false;
            }
            pidx = self.nodes[pidx].next;
        }
        if DEBUG {
            println!("is ear true");
        }
        true
    }

    fn is_ear_hashed(&self, earidx: usize, minx: f32, miny: f32, invsize: f32) -> bool {
		invsize;
        return false;
    } // is ear hashed

    /*
    let a = ear.prev,
        b = ear,
        c = ear.next;

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

    let p = ear.prevz,
        n = ear.nextz;

    // look for points inside the triangle in both directions
    while (p && p.z >= minZ && n && n.z <= maxZ) {
        if (p != ear.prev && p != ear.next &&
            point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) &&
            area(p.prev, p, p.next) >= 0) return false;
        p = p.prevz;

        if (n != ear.prev && n != ear.next &&
            point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, n.x, n.y) &&
            area(n.prev, n, n.next) >= 0) return false;
        n = n.nextz;
    }

    // look for remaining points in decreasing z-order
    while (p && p.z >= minZ) {
        if (p != ear.prev && p != ear.next &&
            point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, p.x, p.y) &&
            area(p.prev, p, p.next) >= 0) return false;
        p = p.prevz;
    }

    // look for remaining points in increasing z-order
    while (n && n.z <= maxZ) {
        if (n != ear.prev && n != ear.next &&
            point_in_triangle(a.x, a.y, b.x, b.y, c.x, c.y, n.x, n.y) &&
            area(n.prev, n, n.next) >= 0) return false;
        n = n.nextz;
    }
    return true;
*/

    // eliminate colinear or duplicate points
    fn filter_points(&mut self, startidx: usize, mut endidx: usize) 
-> usize {
		macro_rules! node {
			($idx:expr) => {{
				&self.nodes[$idx]
			}};
		}
		macro_rules! nodenext {
			($idx:expr) => {{
				&self.nodes[self.nodes[$idx].next]
			}};
		}
		macro_rules! nodeprev {
			($idx:expr) => {{
				&self.nodes[self.nodes[$idx].prev]
			}};
		}

        if startidx == NULL {
            return startidx;
        }
        if endidx == NULL {
            endidx = startidx;
        }

        let mut pidx = 0;
        //let startidx = 0;
        let mut again = true;
        loop {
            again = false;
            if (!(node!(pidx).steiner))
                && (equals( node!(pidx), nodenext!( pidx ) ) 
				|| area(nodeprev!(pidx), node!(pidx), nodenext!(pidx)) == 0.0)
            {
                self.remove_node(pidx);
                endidx = node!(pidx).prev;
                pidx = endidx;
                if pidx == node!(pidx).next {
                    break;
                }
                again = true;
            } else {
                pidx = node!(pidx).next;
            }
            if !(again || pidx != endidx) {
                break;
            }
        }

        return endidx;
    } //filter
} // LL

// create a circular doubly linked list from polygon points in the
// specified winding order
fn linked_list(data: &Vec<f32>, start: usize, end: usize, dim: usize, clockwise: bool) -> LL {
    let mut ll: LL = LL::new();

    if clockwise == (signed_area(&data, start, end, dim) > 0.0) {
        let mut i = start;
        while i < end {
            ll.insert_node(i, data[i], data[i + 1]);
            //ll.dump();
            i += dim;
        }
    } else {
        let mut i = end - dim;
        while i >= start {
            ll.insert_node(i, data[i], data[i + 1]);
            //ll.dump();
            i -= dim;
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
    return (cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0.0
        && (ax - px) * (by - py) - (bx - px) * (ay - py) >= 0.0
        && (bx - px) * (cy - py) - (cx - px) * (by - py) >= 0.0;
}

// return greater of two floating point numbers
fn maxf(a: f32, b: f32) -> f32 {
    if a > b {
        return a;
    }
    return b;
}

pub fn earcut(data: &Vec<f32>, hole_indices: Vec<usize>, ndim: usize) -> Vec<usize> {
    if DEBUG {
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
    ll.dump();
    let mut triangles: Vec<usize> = Vec::new();
    if ll.nodes.len() == 0 {
		if DEBUG { println!("no nodes, triangles: {:?}",triangles); }
        return triangles;
    }

    let (mut minx, mut miny, mut invsize) = (0.0,0.0,0.0);
   	let (mut maxx, mut maxy, mut x, mut y);

    if has_holes {
        ll.eliminate_holes(data, hole_indices, dim);
    }

    // if the shape is not too simple, we'll use z-order curve hash
    // later; calculate polygon bbox
    if DEBUG {
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

        // minx, miny and invsize are later used to transform coords
        // into integers for z-order calculation
        invsize = maxf(maxx - minx, maxy - miny);
        if invsize != 0.0 {
            invsize = 1.0 / invsize;
        } else {
            invsize = 0.0;
        }
    }

    // so basically, for data len < 80*dim, minx,miny are 0
    ll.earcut_linked(&mut triangles, dim, minx, miny, invsize, 0);

    if DEBUG {
        println!("earcut end");
    }
	if DEBUG { println!("triangles: {:?}",triangles); }
    return triangles;
}

// signed area of a triangle
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
        let a = p.prev,
            b = p.next.next;

        if (!equals(a, b) && intersects(a, p, p.next, b) && locallyInside(a, b) && locallyInside(b, a)) {

            triangles.push(a.i / dim);
            triangles.push(p.i / dim);
            triangles.push(b.i / dim);

            // remove two nodes involved
            ll.remove_node(p);
            ll.remove_node(p.next);

            p = start = b;
        }
        p = p.next;
    } while (p != start);

    return p;
}

// try splitting polygon into two and triangulate them independently
fn splitEarcut(start, triangles, dim:usize, minx, miny, invsize) {
    // look for a valid diagonal that divides the polygon into two
    let a = start;
    do {
        let b = a.next.next;
        while (b != a.prev) {
            if (a.i != b.i && isValidDiagonal(a, b)) {
                // split the polygon in two by the diagonal
                let c = splitPolygon(a, b);

                // filter colinear points around the cuts
                a = filter_points(a, a.next);
                c = filter_points(c, c.next);

                // run earcut on each half
                earcut_linked(a, triangles, dim:usize, minx, miny, invsize);
                earcut_linked(c, triangles, dim:usize, minx, miny, invsize);
                return;
            }
            b = b.next;
        }
        a = a.next;
    } while (a != start);
}

fn compareX(a, b) {
    return a.x - b.x;
}

// find a bridge between vertices that connects hole with an outer ring and and link it
fn eliminateHole(hole, outerNode) {
    outerNode = findHoleBridge(hole, outerNode);
    if (outerNode) {
        let b = splitPolygon(outerNode, hole);
        filter_points(b, b.next);
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
        if (hy <= p.y && hy >= p.next.y && p.next.y != p.y) {
            let x = p.x + (hy - p.y) * (p.next.x - p.x) / (p.next.y - p.y);
            if (x <= hx && x > qx) {
                qx = x;
                if (x == hx) {
                    if (hy == p.y) return p;
                    if (hy == p.next.y) return p.next;
                }
                m = p.x < p.next.x ? p : p.next;
            }
        }
        p = p.next;
    } while (p != outerNode);

    if (!m) return NULL;

    if (hx == qx) return m.prev; // hole touches outer segment; pick lower endpoint

    // look for points inside the triangle of hole point, segment intersection and endpoint;
    // if there are no points found, we have a valid connection;
    // otherwise choose the point of the minimum angle with the ray as connection point

    let stop = m,
        mx = m.x,
        my = m.y,
        tanMin = Infinity,
        tan;

    p = m.next;

    while (p != stop) {
        if (hx >= p.x && p.x >= mx && hx != p.x &&
                point_in_triangle(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p.x, p.y)) {

            tan = Math.abs(hy - p.y) / (hx - p.x); // tangential

            if ((tan < tanMin || (tan == tanMin && p.x > m.x)) && locallyInside(p, hole)) {
                m = p;
                tanMin = tan;
            }
        }

        p = p.next;
    }

    return m;
}

// find the leftmost node of a polygon ring
fn getLeftmost(start) {
    let p = start,
        leftmost = start;
    do {
        if (p.x < leftmost.x) leftmost = p;
        p = p.next;
    } while (p != start);

    return leftmost;
}

// check if a diagonal between two polygon nodes is valid (lies in polygon interior)
fn isValidDiagonal(a, b) {
    return a.next.i != b.i && a.prev.i != b.i && !intersectsPolygon(a, b) &&
           locallyInside(a, b) && locallyInside(b, a) && middleInside(a, b);
}


// check if two segments intersect
fn intersects(p1, q1, p2, q2) {
    if ((equals(p1, q1) && equals(p2, q2)) ||
        (equals(p1, q2) && equals(p2, q1))) return true;
    return area(p1, q1, p2) > 0 != area(p1, q1, q2) > 0 &&
           area(p2, q2, p1) > 0 != area(p2, q2, q1) > 0;
}

// check if a polygon diagonal intersects any polygon segments
fn intersectsPolygon(a, b) {
    let p = a;
    do {
        if (p.i != a.i && p.next.i != a.i && p.i != b.i && p.next.i != b.i &&
                intersects(p, p.next, a, b)) return true;
        p = p.next;
    } while (p != a);

    return false;
}

// check if a polygon diagonal is locally inside the polygon
fn locallyInside(a, b) {
    return area(a.prev, a, a.next) < 0 ?
        area(a, b, a.next) >= 0 && area(a, a.prev, b) >= 0 :
        area(a, b, a.prev) < 0 || area(a, a.next, b) < 0;
}

// check if the middle point of a polygon diagonal is inside the polygon
fn middleInside(a, b) {
    let p = a,
        inside = false,
        px = (a.x + b.x) / 2,
        py = (a.y + b.y) / 2;
    do {
        if (((p.y > py) != (p.next.y > py)) && p.next.y != p.y &&
                (px < (p.next.x - p.x) * (py - p.y) / (p.next.y - p.y) + p.x))
            inside = !inside;
        p = p.next;
    } while (p != a);

    return inside;
}

// link two polygon vertices with a bridge; if the vertices belong to the same ring, it splits polygon into two;
// if one belongs to the outer ring and another to a hole, it merges it into a single ring
fn splitPolygon(a, b) {
    let a2 = new Node(a.i, a.x, a.y),
        b2 = new Node(b.i, b.x, b.y),
        an = a.next,
        bp = b.prev;

    a.next = b;
    b.prev = a;

    a2.next = an;
    an.prev = a2;

    b2.next = a2;
    a2.prev = b2;

    bp.next = b2;
    b2.prev = bp;

    return b2;
}
*/

// return a percentage difference between the polygon area and its
// triangulation area; used to verify correctness of triangulation
fn deviation(data: Vec<f32>, hole_indices: Vec<usize>, dim: usize, triangles: Vec<usize>) -> f32 {
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
    let mut i = 0;
    while i < triangles.len() {
        i += 3;
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
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
