
[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)

# Earcutr

This is a port of the MapBox company's Earcut computer code ( see https://github.com/mourner ) , which triangulates polygons. 
Please see https://github.com/mapbox/earcut for more information about the original 
javascript code. This port is to the Rust computer language, and is single-threaded.

This port is not affiliated with MapBox in any way and no endorsement is implied.  Also please note
MapBox has their own Rust port of this code, no claim is made this is better than theirs. 

Please also note someone made this into a Crate (not me) for convenience, please check crates.io

![image showing an outline of a circle with a hole inside of it, with triangles inside of it](viz/circle.png "circle, earcut")

## Usage

```rust
extern crate earcutr;
var triangles = earcutr::earcut(&vec![10,0, 0,50, 60,60, 70,10],&vec![],2);
println!("{:?}",triangles);  // [1, 0, 3, 3, 2, 1]
```

Signature: 

`earcut(vertices:&vec<f64>, hole_indices:&vec<usize>, dimensions:usize)`.

* `vertices` is a flat array of vertex coordinates like `[x0,y0, x1,y1, x2,y2, ...]`.
* `holes` is an array of hole _indices_ if any
  (e.g. `[5, 8]` for a 12-vertex input would mean one hole with vertices 5&ndash;7 and another with 8&ndash;11).
* `dimensions` is the number of coordinates per vertex in the input array. Dimensions must be 2.

Each group of three vertex indices in the resulting array forms a triangle.

```rust
// triangulating a polygon with a hole
earcutr::earcut(&vec![0.,0., 100.,0., 100.,100., 0.,100.,  20.,20., 80.,20., 80.,80., 20.,80.], &vec![4],2);
// [3,0,4, 5,4,0, 3,4,7, 5,0,1, 2,3,7, 6,5,1, 2,7,6, 6,1,2]
```

If you pass a single vertex as a hole, Earcut treats it as a Steiner point. 
See the 'steiner' test under ./tests/fixtures for an example input,
and the test visualization under ./viz.

After getting a triangulation, you can verify its correctness with 
`earcutr.deviation`:

```rust
let deviation = earcutr.deviation(&data.vertices, &data.holes, data.dimensions, &triangles);
```

Deviation returns the relative difference between the total area of 
triangles and the area of the input polygon. `0` means the triangulation 
is fully correct.

## Flattened vs multi-dimensional data

If your input is a multi-dimensional array you can convert it to the 
format expected by Earcut with `earcut.flatten`. For example:

```rust 
let v = vec![
  vec![vec![0.,0.],vec![1.,0.],vec![1.,1.],vec![0.,1.]], // outer ring
  vec![vec![1.,1.],vec![3.,1.],vec![3.,3.]]        // hole ring
];
let (vertices,holes,dimensions) = earcutr::flatten( &v );
let triangles = earcutr::earcut(&vertices, &holes, dimensions);
``` 

The [GeoJSON Polygon](http://geojson.org/geojson-spec.html#polygon) format uses 
multi-dimensional data in a text based JSON format. There is example code under 
tests/integration_test.rs on how to parse JSON data. The test/fixtures test
files are all multi-dimensional .json files.

## How it works: The algorithm

The library implements a modified ear slicing algorithm,
optimized by [z-order curve](http://en.wikipedia.org/wiki/Z-order_curve) hashing
and extended to handle holes, twisted polygons, degeneracies and self-intersections
in a way that doesn't _guarantee_ correctness of triangulation,
but attempts to always produce acceptable results for practical data.

It's based on ideas from
[FIST: Fast Industrial-Strength Triangulation of Polygons](http://www.cosy.sbg.ac.at/~held/projects/triang/triang.html) by Martin Held
and [Triangulation by Ear Clipping](http://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf) by David Eberly.

### Visual example

For example a rectangle could be given in GeoJSON format like so:

    [ [ [0,0],[7,0],[7,4],[0,4] ] ]

This has a single contour, or ring, with four points. The way
the points are listed, it looks 'counter-clockwise' or 'anti-clockwise'
on the page. This is the 'winding' and signifies that it is an 'outer'
ring, or 'body' of the shape.

     _______
     |     |
     |     |
     |     |
     |_____|
 
Now let's add a hole to the square.: 

    [ 
      [ [0,0],[7,0],[7,4],[0,4] ],   
      [ [1,1],[3,1],[3,3] ] 
    ]

This has two contours (rings), the first with four points, the second 
with three points. The second has 'clockwise' winding, signifying it is 
a 'hole'. 

    _______
    |     |
    |  /| |
    | /_| |
    |_____|

After 'flattening', we end up with a single array:

    data [ 0,0,7,0,7,4,0,4,1,1,3,1,3,3  ]
    holeindexes: [ 4 ]
    dimensions: 2

The program will interpret this sequence of data into two separate "rings",
the outside ring and the 'hole'. The rings are stored using a circular
doubly-linked list. 

The program then "removes" the hole, by essentially adding a "cut" between
the hole and the polygon, so that there is only a single "ring" cycle.

         _______
         |     |
         |  /| |
    cut> |_/_| |
         |_____|

The program also automatically 'corrects' the winding of the points 
during input so they are all counter-clockwise.

Then, an "ear cutting" algorithm is applied. But not just any earcutting
algorithm. 

Normally, an ear-cutter algorithm works by finding a potential ear, 
or a "candidate" ear, by looking at three consecutive points on the 
polygon. Then, it must make sure there are no other points "inside"
the ear.

In order to do this, it must iterate through every point in the polygon,
so if your polygon has 15,000 points then it must go through all of them
looking to see if each one is inside the potential ear. Each ear check 
takes a dozen or two calculations, typically using a test like the
wedge product between each side of the ear, and the point to check - if
the point is on the right-hand-side (Wedge is less than zero) of each
side, it's inside the ear, and so the ear cannot be cut. The algorithm
then moves on to the next potential ear.

#### Z-order curve

Z-order hashing allows the number of 'is in ear' checks to be drastically 
cut down. How? Instead of running the "is in ear" code on each other point
in the polygon, it is able to only check points 'nearby' the ear. This is 
accomplished in the following manner:

Step 1: before earcut, each point of the polygon is given a coordinate
on the z-order (Morton) curve through the space of the bounding box 
of the polygon. This is a type of space-filling curve strategy that 
has been explored in many geometry algorithms. Pleas see 
https://en.wikipedia.org/wiki/Z-order_curve

Step 2: The clever bit is that if you want to search within a limited
rectangle inside the space filled by the Z-order curve, you can
relatively easily figure out which points are inside that box by
looking at their position on the z-order curve.. in other words
their z index. The code stores the index as ".z" in each node of the
linked list that represents a vertex of the polygon.

To be more specific, Z-order curves have a special thing about them,
when you pick a limited rectangle inside, you can iterate along the
z-curve from the "lowest" corner to the "highest" corner and be
guaranteed to hit every 2d point inside that rectangle. 

For example, in a 4x4 morton square, there are 16 Morton codes.

     x-----0--1--2--3-->
    y|0    0  1  4  5
     |1    2  3  6  7
     |2    8  9 12 13
     |3   10 11 14 15
    \|/

Going from z-code 0 to z-code 6, 0 1 2 3 4 5 6, takes us through 
every 2 dimensional point between 0,0, where 0 lives, and 2,1, 
where 6 lives. It also goes through 3,0 but nothing is perfect.

Let's say you pick 2,2 and 3,3. The z code at the lowest point
is 12, and the highest is 15. So your z-iteration would be 
12, 13, 14, 15, which covers the rectangle from 2,2 to 3,3.

So, that is how it gets away without checking every point in the polygon
to see if they are inside the ear. It draws a rectangle around the ear,
it finds the lowest and highest corner of that rectangle, and iterates
through the z-curve to check every 2-d point that is 'near' the polygon
ear.

As you can imagine, if 14,000 of your points in your polygon are outside
this box, and only 1000 are in the box, thats quite a bit less math 
and calculation to be done than if you had to iterate through 15,000 points.

#### Additional massaging

If the straightforward earcutting fails, it also does some simple fixes, 

- Filtering points - removing some collinear and equal points

- Self intersections - removing points that only tie a tiny little knot
  in the polygon without contributing to its overall shape (and also
  make it not-simple)

- Split bridge - actually split the polygon into two separate ones,
  and try to earcut each.

Data examples are included under tests/fixtures in json files.
Visualization of test results is generated under viz/testoutput and can
be viewed in a web browser by opening viz/viz.html

### Coordinate number type

The coordinate type in this code is 64-bit floating point. Note that 
32-bit floating point will fail the tests because the test data files 
have numbers that cannot be held with full precision in 32 bits, like 
the base 10 number 537629.886026485, which gets rounded to 537629.875 
during conversion from base 10 to 32-bit base 2.


### Tradeoffs

This triangulator is built primarily as an exercise in porting 
javascript to Rust. However some minor details of the implementation 
have been modified for speed optimization. The code is supposed to 
produce exacly the same output as the javascript version, by using the 
large amount of test data supplied with the original javascript code. 
The speed is comparable with Mapbox's C++ version of earcut, earcut.hpp, 
except for input of very small polygons where the speed is much slower. 
See the benchmarking section for more detail.

If you want to get correct triangulation even on very bad data with lots 
of self-intersections and earcutr is not precise enough, take a look at 
[libtess.js](https://github.com/brendankenny/libtess.js).

You may also want to consider pre-processing the polygon data with 
[Angus J's Clipper](http://angusj.com/delphi/clipper.php) which uses 
Vatti's Algorithm to clean up 'polygon soup' type of data.

### These algorithms are based on linked lists, is that difficult in Rust?

Yes. [A. Beinges's "Too Many Lists"](https://cglab.ca/~abeinges/blah/too-many-lists/book/) 
shows how to do Linked Lists in Rust. Rust also has a 'linked list' type
builtin, which could be made Circular in theory by calling iter().cycle().

However this code implements a Circular Doubly Linked List entirely on 
top of a Rust Vector, without any unsafe blocks. This does not use Rc, 
Box, Arc, etc. The pointers in normal Linked List Node code have been 
replaced by integers which index into a single Vector of Nodes stored in 
LinkedLists struct. It will still crash if you use an index out of bounds
but the RUST_BACKTRACE=1 will tell you exactly where it happened.

This might seem like a source of a slowdown - to bounds check every
array access. However in practice the difference is barely measurable. 
In fact, the code is built so it is relatively easy to switch to
"unchecked_get" to test vector access without bounds checking. 

## Tests, Benchmarks

To run tests:

```bash
$ git clone github.com/donbright/earcutr
$ cd earcutr
$ cargo test             # normal Rust tests. Also outputs visualization data
$ cd viz                 # which is stored under viz/testoutput. you can
$ firefox viz.html       # view in your favorite web browser (circa 2018)
```

To run benchmarks:

```bash
$ cargo bench
...
test bench_water                ... bench:   1,860,385 ns/iter (+/- 21,188)
test bench_water2               ... bench:   1,477,185 ns/iter (+/- 10,294)
test bench_water3               ... bench:      63,800 ns/iter (+/- 3,809)
test bench_water3b              ... bench:       5,751 ns/iter (+/- 18)
test bench_water4               ... bench:     473,971 ns/iter (+/- 5,950)
test bench_water_huge           ... bench:  26,770,194 ns/iter (+/- 532,784)
test bench_water_huge2          ... bench:  53,256,089 ns/iter (+/- 1,208,028)
```

Bench note: As of this writing, benchmarking is not in Stable Rust, so 
this project uses an alternative, https://docs.rs/bencher/0.1.5/bencher/

### Speed of this Rust code vs earcut.hpp C++ code

Following is a rough table based on testing of Earcut's C++ code, 
Earcut's C++ test of LibTess' C++ code, and finally this Rust port of Earcut.

```
____polygon______earcut.hpp_-O2__libtessc++_-O2___Rust_earcutr_release
| water      |  1,831,501 ns/i  |  9,615,384 ns/i |   1,860,385 ns/i |
| water2     |  1,626,016 ns/i  |  1,694,915 ns/i |   1,477,185 ns/i |
| water3     |     53,140 ns/i  |    153,869 ns/i |      63,800 ns/i |
| water3b    |      4,183 ns/i  |     20,143 ns/i |       5,751 ns/i |
| water4     |    475,511 ns/i  |    871,839 ns/i |     473,971 ns/i |
| water_huge | 26,315,789 ns/i  | 26,315,789 ns/i |  26,770,194 ns/i |
| water_huge2| 55,555,555 ns/i  | 20,000,000 ns/i |  53,256,089 ns/i |
----------------------------------------------------------------------
ns/i = nanoseconds per iteration
```

This Rust code appears to be about 20-40% slower than the C++ version of 
Earcut for tiny shapes. However with bigger shapes, it is either within
the error margin, or maybe a bit faster.

#### Method for comparing Earcut C++ / Earcutr Rust

Mapbox has a C++ port of earcut.hpp, with a built in benchmarker, 
measured in 'ops per second'. It also compares against a c++ version of 
libtess. However by default it builds without optimization, which 
hampers comparison. We can fix this.  Editing the .hpp CMakeLists.txt 
file for the C compiler flags lets us turn on optimization,

    add_compile_options("-g" "-O2" ....

Now, Rust bench measures in nanoseconds per iteration.
C++ Earcut measures in iterations per second. To convert:
18 ops in 1 second, is 
18 iterations in 1,000,000,000 nanoseconds. 
1,000,000,000 / 18 -> 55,555,555 nanoseconds/iteration
This way, a conversion can be done. 

#### Profiling

- http://www.codeofview.com/fix-rs/2017/01/24/how-to-optimize-rust-programs-on-linux/

- Valgrind 's callgrind: (see Cargo.toml, set debug=yes)

```bash
sudo apt install valgrind
cargo bench water2 # find the binary name "Running: target/release/..."
valgrind --tool=callgrind target/release/deps/speedtest-bc0e4fb32ac081fc water2
callgrind_annotate callgrind.out.6771
kcachegrind callgrind.out.6771
```

- CpuProfiler 

From AtheMathmo https://github.com/AtheMathmo/cpuprofiler

- Perf

https://perf.wiki.kernel.org/index.php/Tutorial

```bash
cargo bench water2 # find the binary name "Running: target/release/..."
sudo perf stat target/release/deps/speedtest-bc0e4fb32ac081fc  water2
sudo perf record  target/release/deps/speedtest-bc0e4fb32ac081fc  water2
sudo perf report
```

#### Profiling results

Please see [OPTO.md] if you wish a long description of the optimization
process. Here are a few other highlights:

* is_earcut_hashed() is hot

Profilers reveal that on bigger shapes the vast majority of time is 
spent inside is_earcut_hashed(), which is determining whether an ear is 
"really an ear" so that it may be cut without damaging the polygon.

* Zorder is also hot

The second major speed boost comes from Callgrind/kcachegrind in 
particular revealed that the zorder() function was a source of some a 
lot of work by the CPU. In particular the conversion from floating point 
64 bit numbers in the input arguments, to the 32 bit integer, can be 
important to improving speed.

* inline by-hand is important

Most of the time in C++ you can assume the compiler figures out 
inlining. In Rust, however, the point_in_triangle and area function 
inside ear_checker wont get inlined unless specifically indicated with 
the inline macro.

* Vector [Indexing] and bounds checking in Rust doesn't hurt speed here

As mentioned, this code is implemented as a double-linked list sitting on
top of a vector, an the 'pointers' are actually indexes into the vector.
There are several macros used that represent the normal linked list
language, such as next!(index) prev!(index), which take index integers
as input, and return a Node or Reference to Node. Each index uses Rust's
built in vector indexing, which uses 'bounds checking' so it will panic
immediately if memory outside the vector range is accessed on accident.
The panic and backtrace will report exactly what line the access occured
and the value of the index that was too large to use.

Theoretically this slows down the program. In practice, it does not.
This has been tested extensively because the macros like next!() and prev!()
have been written in a way that it is extremely easy to switch back and
forth between bounds-checked vector indexing, and unsafe vector indexing
using get_unchecked(), and re-run 'cargo bench water' to compare them.

The benchmark of water shapes shows the difference is within error, 
except for tiny shapes like water3b, where the benefit is so tiny
as to not be worth it for most usage.

* Iteration vs loops

This code has converted several javascript for loops into Rust 
iteration. In theory this is slower. In practice, it is not, and in some 
cases it is actually faster, especially in find_hole_bridge. In theory
iterators are easier to read and write, take up less code, and have less
bugs.

## This triangulator in other languages

- [mapbox/earcut](https://github.com/mapbox/earcut) MapBox Original javascript
- [mapbox/earcut.hpp](https://github.com/mapbox/earcut.hpp) MapBox C++11


Thanks
