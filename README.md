# Earcutr

This is a single-threaded Polygon triangulation library, translated into 
the Rust computer language from the original javascript code of MapBox's 
Earcut. Please see https://github.com/mapbox/earcut for more information 
about the original javascript code.

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
* `dimensions` is the number of coordinates per vertex in the input array (`2` by default).

Each group of three vertex indices in the resulting array forms a triangle.

```rust
// triangulating a polygon with a hole
earcutr::earcut(&vec![0.,0., 100.,0., 100.,100., 0.,100.,  20.,20., 80.,20., 80.,80., 20.,80.], &vec![4],2);
// [3,0,4, 5,4,0, 3,4,7, 5,0,1, 2,3,7, 6,5,1, 2,7,6, 6,1,2]

// triangulating a polygon with 3d coords
earcutr::earcut(&vec![10.,0.,1., 0.,50.,2., 60.,60.,3., 70.,10.,4.], &vec![], 3);
// [1,0,3, 3,2,1]
```

If you pass a single vertex as a hole, Earcut treats it as a Steiner point. 
See the 'steiner' test under tests/fixtures for an example.

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
    holeindexes: [ 8 ]
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

Then, an "ear cutting" algorithm is applied. But not just any earcutting
algorithm. As alluded to above, it creates a Z-curve in the space of the
polygon, and sorts the points using that Z-curve.

Basically, the linked list has links of 'next' and 'previous' node, 
corresponding to the order given and processed by the main preparatory
algorithm. But then each node also has a "next Z" and "prev Z" node,
so that the list allows the program to quickly check whether points
are Ears or not by looking "nearby" it first. 

"Is Ear" is the key loop of any Earclip algorithm, it may 
run this one function millions of times on a shape with a few thousand
points.

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
javascript to Rust. It is supposed to produce exacly the same output as 
the javascript version, thanks to the test data supplied with the 
original javascript code.. It is relatively simple. If the benchmarks 
below are correct, the optimized Rust speed is a wee bit slower
than the optimized C++ version of earcut, but usually in the same ballpark.

If you want to get correct triangulation even on very bad data with lots 
of self-intersections and earcutr is not precise enough, take a look at 
[libtess.js](https://github.com/brendankenny/libtess.js).

You may also want to consider pre-processing the polygon data with 
[Angus J's Clipper](http://angusj.com/delphi/clipper.php) which uses 
Vatti's Algorithm to clean up 'polygon soup' type of data.

### These algorithms are based on linked lists, is that difficult in Rust?

Yes. [A. Beinges's "Too Many Lists"](https://cglab.ca/~abeinges/blah/too-many-lists/book/) 
shows how to do Linked Lists in Rust.

However this code implements a Circular Doubly Linked List entirely on 
top of a Rust Vector, so that there is no unsafe code, and no reference 
cycles. This does not use Rc, Box, Arc, etc. The pointers in normal 
Linked List Node code have been replaced by integers which index into a 
single Vector of Nodes stored in LinkedLists struct.

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
test bench_water                ... bench:   1,996,418 ns/iter (+/- 55,718)
test bench_water2               ... bench:   1,594,661 ns/iter (+/- 12,079)
test bench_water3               ... bench:      72,284 ns/iter (+/- 537)
test bench_water3b              ... bench:       8,218 ns/iter (+/- 40)
test bench_water4               ... bench:     508,055 ns/iter (+/- 3,053)
test bench_water_huge           ... bench:  30,273,199 ns/iter (+/- 629,866)
test bench_water_huge2          ... bench:  61,670,582 ns/iter (+/- 1,541,937)
```

Bench note: As of this writing, benchmarking is not in Stable Rust, so 
this project uses an alternative, https://docs.rs/bencher/0.1.5/bencher/

### Speed of this Rust code vs earcut.hpp C++ code

Mapbox has a C++ port of earcut.hpp, with a built in benchmarker, measured
in 'ops per second'. For water tests, it reports like so on an old HP Laptop,
without any C++ optimizations ( -O0 ).

```bash
____polygon_________________earcut.hpp_________libtessc++___
| water          |          231 ops/s |           70 ops/s |
| water2         |          183 ops/s |          325 ops/s |
| water3         |        4,198 ops/s |        2,558 ops/s |
| water3b        |       41,345 ops/s |       15,412 ops/s |
| water4         |          784 ops/s |          593 ops/s |
| water_huge     |           19 ops/s |           27 ops/s |
| water_huge2    |            8 ops/s |           36 ops/s |
------------------------------------------------------------

Now... you can hack around in the earcut.hpp CMakeLists.txt cmakefile, 
and Turn On Optimization (gcc/clang -O2). It becomes at least a 2x speedup,
sometimes much higher.

____polygon_________________earcut.hpp_________libtessc++___
| water          |          546 ops/s |          104 ops/s |
| water2         |          615 ops/s |          590 ops/s |
| water3         |       18,818 ops/s |        6,499 ops/s |
| water3b        |      239,026 ops/s |       49,645 ops/s |
| water4         |        2,103 ops/s |        1,147 ops/s |
| water_huge     |           38 ops/s |           38 ops/s |
| water_huge2    |           18 ops/s |           50 ops/s |
------------------------------------------------------------

Now, Rust bench measures in nanoseconds per iteration.
C++ Earcut measures in iterations per second. To convert:
18 ops in 1 second, is 
18 iterations in 1,000,000,000 nanoseconds. 
1,000,000,000 / 18 -> 55,555,555 nanoseconds/iteration
So, converting the above:

____polygon______earcut.hpp_-O2__libtessc++_-O2___Rust_earcutr_release
| water      |  1,831,501 ns/i  |  9,615,384 ns/i |   1,996,418 ns/i |
| water2     |  1,626,016 ns/i  |  1,694,915 ns/i |   1,594,661 ns/i |
| water3     |     53,140 ns/i  |    153,869 ns/i |      88,000 ns/i |
| water3b    |      4,183 ns/i  |     20,143 ns/i |       8,218 ns/i |
| water4     |    475,511 ns/i  |    871,839 ns/i |     508,055 ns/i |
| water_huge | 26,315,789 ns/i  | 26,315,789 ns/i |  30,273,199 ns/i |
| water_huge2| 55,555,555 ns/i  | 20,000,000 ns/i |  61,670,582 ns/i |
----------------------------------------------------------------------
ns/i = nanoseconds per iteration

```

If the calculations are correct in the Rust benchmark, and the 
conversion is correct, then the Rust code is 10 to 20% slower than the 
C++ port of earcut.hpp on most shapes. 

#### Profiling

- http://www.codeofview.com/fix-rs/2017/01/24/how-to-optimize-rust-programs-on-linux/

- Valgrind 's callgrind: (see Cargo.toml, set debug=yes)

```bash
sudo apt install valgrind
cargo bench dude # find the binary name "Running: target/release/..."
valgrind --tool=callgrind target/release/deps/speedtest-bc0e4fb32ac081fc dude
callgrind_annotate callgrind.out.6771
kcachegrind callgrind.out.6771
```

- CpuProfiler 

From AtheMathmo https://github.com/AtheMathmo/cpuprofiler

- Perf

https://perf.wiki.kernel.org/index.php/Tutorial

```bash
sudo perf stat target/release/deps/speedtest-bc0e4fb32ac081fc  dude
sudo perf record  target/release/deps/speedtest-bc0e4fb32ac081fc  dude
sudo perf report
```

#### Profiling results 1/2 - Rust specific

Here are a few other highlights from testing:

* is_earcut_hashed() is hot: Profilers reveal that on bigger shapes the vast majority of time is spent 
inside is_earcut_hashed(), which is determining whether an ear is 
"really an ear" so that it may be cut without damaging the polygon.

* inline is important: Most of the time in C++ you can assume the 
compiler figures out inlining. Here, however, the point_in_triangle and 
area function take up a lot of time. Since this port has them inside a 
single 'earchecker' function, that function was marked 'inline' 
resulting in a good speed boost.

* Zorder is also hot: The second major speed boost comes from 
Callgrind/kcachegrind in particular revealed that the zorder() function 
was a source of some consternation. In particular the conversion from 
floating point 64 bit numbers in the input arguments, to the 32 bit 
integer, can be tricky since the conversion can be optimized in various 
unusual ways

* Floating point to integer is hot:

By transforming 

    let mut x: i32 = 32767 * ((xf- minx) * invsize).round() as i32;
    let mut y: i32 = 32767 * ((yf- miny) * invsize).round() as i32;

Into this:

    let mut x: i32 = ( 32767.0 *   ((xf - minx) * invsize)) as i32;
    let mut y: i32 = ( 32767.0 *   ((yf - miny) * invsize)) as i32;

A 13x speedup was achieved for the 'water' benchmark.

* dimensions: c++ earcut assumes 2 dimensions, which can save a few percent. for example
the main earcut loop, which is called for every single ear, has three 'div'
inside of it if you have generic dimensions. by optimizing for the case
of 2 dimensions by saying point.i/2, the compiler can change that into
a >> 2

* linked list vs nodes in a vector: c++ earcut uses 'real linked lists' 
instead of this code, which fakes a linked list using bounds-checked 
index into a vector of nodes. some of this can be replaced with 
get_unchecked but experiments only showed a tiny speed boost within the 
margin of error, except for small shapes like water3b which was about 
500 ns, but not worth an unsafe{} block. during development the program
has crashed many times due to out of bounds checking, and thanks to that
checking it was extremely easy to diagnose and fix the problems. unsafe
would remove that ability.

* NULL index vs Option(Index). Since we use a vector of nodes instead
of a linked list, there needs to be a translation of the NULL concept
like in the javascript code for the linked list node pointers. We could
use 'Option' but what ends up happening is that you have to unwrap
your option every single time. And what risk is it stopping? A bad index
into a vector - which will get caught by the bounds checker if its too
low or high, or worst case produce garbled output by going to invalid
nodes. I tried to convert NULL to Option None, everything was about
twice as slow in benchmarks.

* linked list 'next' vs iterator: this version of rust code uses an 
iterator to cycle through the nodes, which is slightly slower than a 
real linked list since the custom iterator is checking 4 conditions 
instead of 1 (is null)

* small shapes: for water3b in particular, the rust code is spending 
more time in earcut_linked than in is_ear, which is opposite from c++

* maths: there is a small possibility that c++ earcut appears to be 
optimizing the point_in_tria
ngle math differently than Rust,

* Subtraction that is unnoticeable: what about eliminating the 
'subtraction' in point_in_triangle by translating the entire polygon so 
minx and miny are 0? Tried it, the difference is within margin of 
measurement error. In other words, even though its removing millions of 
instructions from the code line, the way things fit through the CPU 
floating point cache it didnt matter. Amazing.

* Vector Indexing and bounds checking in Rust: You can test this by 
replacing [] inside of the node! next! and prev! macros with 
.get_unchecked() and/or .get_unchecked_mut(), the answer is a tiny 
speedup, almost too small to measure.

* Iteration vs loops. This code has converted several javascript for 
loops into Rust iteration. For example the entire leftmost() function 
was replaced by a single line of iterator adaptor code. However. Each 
iteration involves calling Some() and checking against None which may be 
slower than doing a straight and ordinary loop. How much slower?

```
a few iter_range() replaced with loop{}: 
test bench_water                ... bench:   2,044,420 ns/iter (+/- 23,388)
test bench_water2               ... bench:   1,865,605 ns/iter (+/- 13,123)
test bench_water3               ... bench:      77,158 ns/iter (+/- 483)
test bench_water3b              ... bench:       6,845 ns/iter (+/- 35)
test bench_water4               ... bench:     530,067 ns/iter (+/- 14,930)
test bench_water_huge           ... bench:  30,693,084 ns/iter (+/- 578,848)
test bench_water_huge2          ... bench:  62,481,656 ns/iter (+/- 1,934,261)

```

Ahh. Well. For testing, a few iterations were replaced with loop{}. the 
small shape water3b it is several hundred nanoseconds faster per 
iteration using loop{}. However for larger shapes the speed is within 
error of measurement. 

Furthermore, consider maintainability, legibility, likelihood of bugs, 
and elegance. Let's compare is_ear() which is used heavily by small 
shapes (since larger shapes only use is_ear_hashed()). Compare the loop{}
version with the iteration version:

```rust

Loop: 10 lines of code. Normal C style maintainability, and bug 
issues (off by one in the loop? breaking too soon or too late? not 
cleaning up properly before return? not a big deal, but can we do 
better?)

            let mut p = c.next_idx;
            loop {
                if point_in_triangle(&a,&b,&c,&node!(ll,p)) &&
                area(&prev!(ll, p), &node!(ll,p), &next!(ll, p)) >= 0.0 {
                    return false;
                }
                p = node!(ll,p).next_idx;
                if p==a.idx { break; };
            };
            return true;

Iteration: 4 lines. Easier to read (once you learn the basics of 
functional languages), easier to maintain, easier to debug, easier to 
write, "off by one" only applies to my iteration inputs c.next and a, 
and we don't have to manually break or return at all.

          !ll.iter_range(c.next_idx..a.idx).any(|p| {
            point_in_triangle(&a, &b, &c, &p)
                && (area(&prev!(ll, p.idx), &p, &next!(ll, p.idx)) >= 0.0)
		  }),

```

Taking into consideration the balance and tradeoff between speed and 
maintainability, legibility, and elegance, iterators have been left in. 
The speed is unnoticeable on the vast majority of benchmarks and it's 
only a tiny percentage on others.

Note that part of this is because there is no place for iterator to be 
used in the 'hot code', where most time is spent - is_ear_hashed(). 

#### Profiling results 2/2 - special sauce

Profiling above showed the zorder() function is hot. Can we cool it
down a bit?

It begins like this:

```rust
fn zorder(xf: f64, yf: f64, minx: f64, miny: f64, invsize: f64) -> i32 {
    // coords are transformed into non-negative 15-bit integer range
    let mut x: i32 = (32767.0 * ((xf - minx) * invsize)) as i32;
    let mut y: i32 = (32767.0 * ((yf - miny) * invsize)) as i32;
```

Now, I find its easier to read the assembly by actually using 'stat'
than making rust dump assembly, so trying this:

```bash

make sure Cargo.toml has

[profile.release] 
debug = true

then

don@serebryanya:~/src/earcutr$ cargo bench water2
   Compiling earcutr v0.1.0 (/home/don/src/earcutr)                             
    Finished release [optimized + debuginfo] target(s) in 9.11s                 
     Running target/release/deps/earcutr-7c59a70327baf37c
     Running target/release/deps/speedtest-c6de42429f2eb5ef
don@serebryanya:~/src/earcutr$ sudo perf stat target/release/deps/speedtest-c6de42429f2eb5ef water2
don@serebryanya:~/src/earcutr$ sudo perf record target/release/deps/speedtest-c6de42429f2eb5ef water2
don@serebryanya:~/src/earcutr$ sudo perf report

hit 'h' to learn about keyboard commands
drill down into is_earcut_hashed
scroll to the top
hit / for search, type 'invsize'

       │     _ZN7earcutr13is_ear_hashed17haea87f38a9e0c9d3E():                 ▒
  0.02 │       movsd  0x61056(%rip),%xmm8        # 82ea0 <ryu::d2s_full_table::▒
       │     _ZN7earcutr6zorder17hab1d711e708e67cdE():                         ▒
       │         let mut x: i32 = (32767.0 * ((xf - minx) * invsize)) as i32;  ▒
       │       subsd  %xmm0,%xmm6                                              ▒
       │       mulsd  %xmm2,%xmm6                                              ▒
  0.69 │       mulsd  %xmm8,%xmm6                                              ▒
  0.63 │       cvttsd2si %xmm6,%eax                                            ▒
       │         let mut y: i32 = (32767.0 * ((yf - miny) * invsize)) as i32;  ▒
  0.02 │       subsd  %xmm1,%xmm5                                              ▒
  0.54 │       mulsd  %xmm2,%xmm5                                              ▒
  0.31 │       mulsd  %xmm8,%xmm5                                              ▒
  0.27 │       cvttsd2si %xmm5,%esi

```

What is this saying? Googleing subsd, mulsd, and cvttsd2si shows this:

Basically it is subtracting one float from another, then multiplying 
it, then multiplying it again. Then it's rounding to integer.

But what is 32767.0 all about? invsize is multiplied by it... twice...
couldn't we pre-computer invsize*32767.0 once instead of twice?

Well maybe the optimizer has some reason. But ... what if we never
had to multiply it at all? invsize is hardly ever used. In fact,
pulling out trust nano 'ctrl-w' invsize, it reveals that invsize
is never ever used except inside zorder, right here in the hot zone
that im looking at!

Now... where is invsize really set up? It's in calc_invsize(), note
that i heavily modded that function while porting the javascript to try to
understand what was going on better.

```rust
// into integers for z-order calculation
fn calc_invsize(minx: f64, miny: f64, maxx: f64, maxy: f64) -> f64 {
    let invsize = f64::max(maxx - minx, maxy - miny);
    match invsize == 0.0 {
        true => 0.0,
        false => 1.0 / invsize,
    }
}
```

So invsize is actually 1/ itself. But ok. What if we pre-multiply by 
32767.0 right here?

```
// into integers for z-order calculation
fn calc_invsize(minx: f64, miny: f64, maxx: f64, maxy: f64) -> f64 {
    let invsize = f64::max(maxx - minx, maxy - miny);
    match invsize == 0.0 {
        true => 0.0,
        false => 32767.0 / invsize,
    }
}
```

Now what does the ASM become?

```asm

       │     _ZN7earcutr6zorder17hab1d711e708e67cdE():                         ▒
       │         let mut x: i32 = ( ((xf - minx) * invsize)) as i32;           ▒
       │       subsd  %xmm0,%xmm6                                              ▒
       │       mulsd  %xmm2,%xmm6                                              ▒
  0.71 │       cvttsd2si %xmm6,%eax                                            ▒
       │         let mut y: i32 = ( ((yf - miny) * invsize)) as i32;           ▒
  0.26 │       subsd  %xmm1,%xmm5                                              ▒
  0.19 │       mulsd  %xmm2,%xmm5                                              ▒
  0.97 │       cvttsd2si %xmm5,%esi
```

Wow... i got rid of a whole instruction inside a hot zone. What are the 
results? First, cargo test showed there were no test failures. Then, 
cargo bench water:


```bash
test bench_water                ... bench:   2,054,921 ns/iter (+/- 31,695)
test bench_water2               ... bench:   1,854,223 ns/iter (+/- 227,359)
test bench_water3               ... bench:      76,377 ns/iter (+/- 595)
test bench_water3b              ... bench:       7,053 ns/iter (+/- 17)
test bench_water4               ... bench:     530,306 ns/iter (+/- 24,262)
test bench_water_huge           ... bench:  30,679,819 ns/iter (+/- 588,042)
test bench_water_huge2          ... bench:  62,861,730 ns/iter (+/- 1,542,146)
```

Well, not great, its well within margin of error. But i'll take it. 
Unlike optimizations above which reduced safety or took up more code, 
this is actually reducing code by a few symbols. I like that type of 
optimization better than removing safety.


* translate to zero

The zorder also has 'xf - minx'. What is Minx? Its the 'left most' point
on the polygon input data. But why do we have to have it? Can we translate
the whole polygon so minx is 0? Then we don't need this subtraction.

Abit of work and, again perf reveals another instruction can be eliminated
from zorder, the first two lines are just a single multiplication now:

```asm
       │     _ZN7earcutr6zorder17hf6b08d8ab5a82e32E():                         ◆
       │         let mut x: i32 = ( ((xf ) * invsize)) as i32;                 ▒
  0.35 │       mulsd  %xmm0,%xmm3                                              ▒
  0.09 │       cvttsd2si %xmm3,%eax                                            ▒
       │     _ZN4core3f6421_$LT$impl$u20$f64$GT$3max17h0592a962afedea39E():    ▒
  0.24 │       orpd   %xmm4,%xmm2                                              ▒
       │     _ZN7earcutr6zorder17hf6b08d8ab5a82e32E():                         ▒
       │         let mut y: i32 = ( ((yf ) * invsize)) as i32;                 ▒
  0.50 │       mulsd  %xmm0,%xmm2                                              ▒
  0.39 │       cvttsd2si %xmm2,%esi 
```

not even sure what 'orpd' is doing in there... apparently the optimizer
felt it could slip it in there without a problem 

Benchmark results? Well tbh no visible change. It proves that this
eliminating instructions is not always going to save  time in the running.

But can we go further?

Look at the zorder function.. it's using 32 bit numbers and shifting.

```rust 
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;
```

What if i combined these?

```rust 

    let mut x: i64 = ( ((xf ) * invsize)) as i64;
    let mut y: i64 = ( ((yf ) * invsize)) as i64;
    let mut xy:i64 = x << 32 | y;

    xy = (xy | (xy << 8)) & 0x00FF00FF00FF00FF;
    xy = (xy | (xy << 4)) & 0x0F0F0F0F0F0F0F0F;
    xy = (xy | (xy << 2)) & 0x3333333333333333;
    xy = (xy | (xy << 1)) & 0x5555555555555555;

    let x32: i32 =(  xy >> 32 ) as i32;
    let y32: i32 =(  xy & 0x00000000FFFFFFFF ) as i32;

```


```asm
  0.09 │       mov    %rax,%rsi                                                ▒
  0.46 │       shl    $0x8,%rsi                                                ▒
  0.70 │       or     %rax,%rsi                                                ▒
  0.29 │       and    %rbx,%rsi                                                ▒
       │         xy = (xy | (xy << 4)) & 0x0F0F0F0F0F0F0F0F;                   ◆
  0.04 │       mov    %rsi,%rax                                                ▒
  0.26 │       shl    $0x4,%rax                                                ▒
  0.70 │       or     %rsi,%rax                                                ▒
  0.13 │       and    %r14,%rax                                                ▒
       │         xy = (xy | (xy << 2)) & 0x3333333333333333;                   ▒
  0.31 │       lea    0x0(,%rax,4),%rsi                                        ▒
  0.40 │       or     %rax,%rsi                                                ▒
  0.29 │       and    %r11,%rsi                                                ▒
       │         xy = (xy | (xy << 1)) & 0x5555555555555555;                   ▒
  0.33 │       lea    (%rsi,%rsi,1),%rax                                       ▒
  0.24 │       or     %rsi,%rax                                                ▒
       │             let x32: i32 =(  xy >> 32 ) as i32;                       ▒
  0.02 │       mov    %rax,%rsi                                                ▒
  0.24 │       shr    $0x20,%rsi                                               ▒
  0.57 │       and    $0x55555555,%esi                                         ▒
       │         x32 | (y32 << 1)                                              ▒
       │       and    $0x55555555,%eax
```

```bash
test bench_water                ... bench:   2,056,546 ns/iter (+/- 31,190)
test bench_water2               ... bench:   1,789,347 ns/iter (+/- 11,558)
test bench_water3               ... bench:      78,216 ns/iter (+/- 637)
test bench_water3b              ... bench:       7,170 ns/iter (+/- 59)
test bench_water4               ... bench:     522,485 ns/iter (+/- 5,929)
test bench_water_huge           ... bench:  30,443,739 ns/iter (+/- 730,965)
test bench_water_huge2          ... bench:  61,979,392 ns/iter (+/- 1,704,953)
```

again, not great, but i'll take it.

* freelist

I thought i would be clever and during remove_node, create a list of
removed nodes. Turns out, even though we are faking a linked list
inside a vector... we dont need a freelist of removed nodes. Except
for debugging and development when dumping the internal structure of
the nodes to printout, we dont need freelist at all. Removing it saved
a few percent.

* NULL

For Null pointer, well, I made an index of 0x7774917.. That's "magic" in
l33tspeak. However. Internal CPU sometimes prefer numbers like 0 or
FFFFFFFFFF to compare bits. Changing this to ffffffff... brought another
few percent.

```bash
test bench_water                ... bench:   2,033,630 ns/iter (+/- 40,087)
test bench_water2               ... bench:   1,778,138 ns/iter (+/- 133,746)
test bench_water3               ... bench:      74,565 ns/iter (+/- 695)
test bench_water3b              ... bench:       6,588 ns/iter (+/- 68)
test bench_water4               ... bench:     514,063 ns/iter (+/- 2,866)
test bench_water_huge           ... bench:  30,305,755 ns/iter (+/- 749,417)
test bench_water_huge2          ... bench:  61,937,131 ns/iter (+/- 2,259,247)
```

* the main earcut loop

This is the 'next most' hot section after is_earcut_hashed and zorder.
Why is it so long though? It has numerous ifs, thens, elses. Can it
be simplified?

It turns out, yes. One optimization of C++ earcut is that if there are
more than 80 points, they do not use the hashed_earcut. This results in 
if then else. What if we just ignore that and always use the hash
version? For tiny shapes like water3b we lose a little speed, but for
big shapes ... we gain.

```bash
test bench_water                ... bench:   1,979,939 ns/iter (+/- 43,625)
test bench_water2               ... bench:   1,686,064 ns/iter (+/- 11,793)
test bench_water3               ... bench:      88,216 ns/iter (+/- 14,474)
test bench_water3b              ... bench:       6,393 ns/iter (+/- 27)
test bench_water4               ... bench:     497,426 ns/iter (+/- 17,692)
test bench_water_huge           ... bench:  30,083,359 ns/iter (+/- 746,826)
test bench_water_huge2          ... bench:  61,794,920 ns/iter (+/- 2,600,562)
```

For some shapes we are now within a percent of C++. 

* NULL Node

Instead of NULL being ffffffff.. what if it was 000000.. and what if...
dereferencing a null pointer to a node was ... well.. perfectly valid?
What if there was, in fact, an object at address 0x00000000.. in our case
we are using a vector, not RAM itself, so we can even put an object
at 0x000000000. We can put a node there. We can call it the 'null node'.
It is just like any other node, except that it is never part of a ring.

What does this do? Well, it eliminates two branch conditions from
remove_node()!

        let prz = node!(self, p).prevz_idx;
        let nxz = node!(self, p).nextz_idx;
		if prz!=NULL {nodemut!(self, prz).nextz_idx = nxz;}
        if prz!=NULL {nodemut!(self, nxz).prevz_idx = prz;}

becomes

        let prz = node!(self, p).prevz_idx;
        let nxz = node!(self, p).nextz_idx;
		nodemut!(self, prz).nextz_idx = nxz;
        nodemut!(self, nxz).prevz_idx = prz;

So if someone removes a node adjacent to the null node, the null node's
nextz_idx and prevz_idx are modified... but it doesnt matter because
they are never used.


## This triangulator in other languages

- [mapbox/earcut](https://github.com/mapbox/earcut) MapBox Original javascript
- [mapbox/earcut.hpp](https://github.com/mapbox/earcut.hpp) MapBox C++11


Thanks
