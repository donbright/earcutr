## Earcutr

Warning: This is a work in progress, not 100% equivalent to earcut.js 
yet, it needs some speed improvement, however only one test fails as of 
this writing.

Polygon triangulation library, translated into Rust computer language from
the original Earcut project from MapBox. https://github.com/mapbox/earcut

![image showing an outline of a building, with triangles inside of it](viz/circle.png "circle, earcut")

#### Usage

```rust
var triangles = earcut([10,0, 0,50, 60,60, 70,10]);
```

Signature: `earcut(vertices[, holes, dimensions = 2])`.

* `vertices` is a flat array of vertex coordinates like `[x0,y0, x1,y1, x2,y2, ...]`.
* `holes` is an array of hole _indices_ if any
  (e.g. `[5, 8]` for a 12-vertex input would mean one hole with vertices 5&ndash;7 and another with 8&ndash;11).
* `dimensions` is the number of coordinates per vertex in the input array (`2` by default).

Each group of three vertex indices in the resulting array forms a triangle.

```rust
// triangulating a polygon with a hole
earcut([0,0, 100,0, 100,100, 0,100,  20,20, 80,20, 80,80, 20,80], [4]);
// [3,0,4, 5,4,0, 3,4,7, 5,0,1, 2,3,7, 6,5,1, 2,7,6, 6,1,2]

// triangulating a polygon with 3d coords
earcut([10,0,1, 0,50,2, 60,60,3, 70,10,4], null, 3);
// [1,0,3, 3,2,1]
```

If you pass a single vertex as a hole, Earcut treats it as a Steiner point.

If your input is a multi-dimensional array (e.g. [GeoJSON Polygon](http://geojson.org/geojson-spec.html#polygon)),
you can convert it to the format expected by Earcut with `earcut.flatten`:

```rust
let v = vec![vec![vec![0.0,0.0],vec![1.0,0.0],vec![1.0,1.0],vec![0.0,1.0]]];
let holes:Vec<usize> = vec![];
let data = earcutr.flatten( v );
let triangles = earcut(&data.vertices, &data.holes, data.dimensions);
```

After getting a triangulation, you can verify its correctness with 
`earcutr.deviation`:

```rust
let deviation = earcutr.deviation(&data.vertices, &data.holes, data.dimensions, &triangles);
```

Returns the relative difference between the total area of triangles and 
the area of the input polygon. `0` means the triangulation is fully 
correct.

#### How it works: The algorithm

The library implements a modified ear slicing algorithm,
optimized by [z-order curve](http://en.wikipedia.org/wiki/Z-order_curve) hashing
and extended to handle holes, twisted polygons, degeneracies and self-intersections
in a way that doesn't _guarantee_ correctness of triangulation,
but attempts to always produce acceptable results for practical data.

It's based on ideas from
[FIST: Fast Industrial-Strength Triangulation of Polygons](http://www.cosy.sbg.ac.at/~held/projects/triang/triang.html) by Martin Held
and [Triangulation by Ear Clipping](http://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf) by David Eberly.

#### Visual example

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
 
A square with a triangle shaped hole in the middle would be as follows

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

Notice we can still interpret the data as a sequence of points,
and by looking at 'hole indexes' we can figure out where the first contour
ends and the next begins. If we had two holes, there'd be two holeindexes.

Data examples are included under tests/fixtures in json files.

#### Why another triangulation library?

This is for speed, simplicity, small size, and to make a port of earcut.js
to Rust.

If you want to get correct triangulation even on very bad data with lots of self-intersections
and earcutr is not precise enough, take a look at [libtess.js](https://github.com/brendankenny/libtess.js).

Or pre-process the data with [Angus J's 
Clipper](http://angusj.com/delphi/clipper.php) which uses Vatti's 
Algorithm to clean up 'polygon soup' type of data.

#### Install

You can copy the earcutr.rs file into your own project and use it.

To download the full library, with tests,

```bash
git clone github.com/donbright/earcutr
cd earcutr
cargo test                      # normal build and test report
cargo test -- --test-threads=1  # test-threads=1 will create visualization data
ls viz/testoutput.json # if visualization worked, this file will be created
cd viz                 # vizualisation code lives here, it's javascript/html
firefox viz.html       # view in your favorite web browser (circa 2018)
```

#### Ports to other languages

- https://github.com/mapbox/earcut (Original javascript, earcutr is a port)
- [mapbox/earcut.hpp](https://github.com/mapbox/earcut.hpp) (C++11)
