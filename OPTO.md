
#### Optimization

After initial porting of javascript to Rust, the code was quite slow.
About ten times slower. 

After much messing around with the C++ port of earcut, it was clear
the C++ version was extremely fast. The benchmarks even had optimization
turned off. Turning it on sometimes was 10x speedup or more.

Through profiling and some optimization work, improvements were made
to the Rust code so that it was comparable to optimized C++.

Here are some highlights of that process, along with the benchmarks of
the water* shapes as the code slowly gets faster and faster. 


## C++ vs javascript

The C++ version made a few decisions that by default made it faster than
javascript:

* C++ only accepts two dimensions

This eliminates several "div" (division) from the main hot loop which is
executed millions of times for a good sized shape.



## Rust vs C++
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

## Profiling results 2/2 - special sauce

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

Wow... i got rid of a whole mulsd instruction inside a hot zone. What are the 
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

In which we eliminate "is this a pointer to NULL" branch conditional in
several pieces of linked list code.

Instead of NULL being ffffffff.. what if it was 000000.. and what if...
dereferencing a null pointer to a node was ... well.. perfectly valid?
What if there was, in fact, an object at address 0x00000000.. in our case
we are using a vector, not RAM itself, so we can even put an object
at 0x000000000. We can put a node there. We can call it the 'null node'.
It is just like any other node, except that it is never part of a ring.

Null is typically 'bad' in that it's encoded as a pointer to the memory
address of 0x000000000. Computers don't like people poking around
at address 0, so they typically stop the program (Segfault / crash) if
your program tries to do it.

Here however, we dont have pointers to nodes. We have indexes into a Vector.
The 'null pointer' is still 0.. but we can actually stick an extra
node in at 0 so that the program wont crash when dereferencing.

Why? Well, a normal C++ program or Javascript program will have a lot of
conditional branches and if statements to make sure a pointer isnt zero
before accessing whatever it points to, so to avoid crashing.

But those branches make the program slower. CPUs are designed to be fastest
without branches. So the null node allows us to eliminate branches.
We dont care if remove_node sets the NULL node's next_index to a number,
so we dont have to have a conditional branch inside remove_node

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


* NULL node optimization in is_ear_hashed

```rust
    while p!=NULL && node!(ll, p).z >= min_z {
        if earcheck(&a,&b,&c,prevref!(ll, p),noderef!(ll, p),nextref!(ll, p),
        ) {
            return false;
        }
        p = node!(ll, p).prevz_idx;
    }
```

can be changed, eliminating a conditional check that is computed
millions of times for a polygon with a few thousand points. Here is how:

```rust
    nodemut!(ll, NULL).z = min_z - 1;
    while node!(ll, p).z >= min_z {
        if earcheck(&a,&b,&c,prevref!(ll, p),noderef!(ll, p),nextref!(ll, p),
        ) {
            return false;
        }
        p = node!(ll, p).prevz_idx;
    }
```

What we did was allow the code to dereference the NULL 'pointer', since 
NULL is not a pointer, it's an index into a Vector. And we have put a 
'null node' at vector position 0, so vec[NULL] becomes vec[0] which is 
totally valid. This eliminates a branch which allows the CPU to go 
faster, as it only has one thing to check now, not two.

* iterator optimization

In which we change complicated for loops into chained iteration with
filters, folds, and other functional language features.

My old iterator, first attempt, did several conditional checks per loop,
it was checking if it was done, if it had reached a null node, if it
reached the end, if it has a 'single node' chain, etc etc.

i changed it to preload itself with data, called 'pending_result', 
then on each iteration it returns the pending result, and increments
the 'current node' to the next node. now we only have one conditional - 
if current is end, make the 'pending result' None.


* loop into iterator

after optimizing my iterator as above, i investigated some of the
loop vs iterator tests i did before:


```
fn intersects_polygon(ll: &LinkedLists, a: &Node, b: &Node) -> bool {
    let mut p = a.idx;
    loop {
        if noderef!(ll, p).i != a.i
            && next!(ll, p).i != a.i
            && noderef!(ll, p).i != b.i
            && next!(ll, p).i != b.i
            && pseudo_intersects(noderef!(ll, p), nextref!(ll, p), a, b)
        {
            return true;
        }
        p = next!(ll, p).idx;
        if p == a.idx {
            break;
        };
   }
   return false
```

```bash
running 7 tests
test bench_water                ... bench:   1,990,314 ns/iter (+/- 44,007)
test bench_water2               ... bench:   1,503,751 ns/iter (+/- 180,270)
test bench_water3               ... bench:      70,347 ns/iter (+/- 469)
test bench_water3b              ... bench:       7,712 ns/iter (+/- 27)
test bench_water4               ... bench:     499,915 ns/iter (+/- 17,212)
test bench_water_huge           ... bench:  30,275,308 ns/iter (+/- 659,648)
test bench_water_huge2          ... bench:  62,049,947 ns/iter (+/- 1,785,038
```

my iterator version is now, actually, the same speed basically, but
a bit less code and easier to read( if you know functional program basics)

```rust
        ll.iter_range(a.idx..a.idx).any(|p| {
        p.i != a.i
            && next!(ll, p.idx).i != a.i
            && p.i != b.i
            && next!(ll, p.idx).i != b.i
            && pseudo_intersects(&p, nextref!(ll, p.idx), a, b)
    })
```

```bash
running 7 tests
test bench_water                ... bench:   1,972,265 ns/iter (+/- 63,654)
test bench_water2               ... bench:   1,496,676 ns/iter (+/- 19,040)
test bench_water3               ... bench:      69,862 ns/iter (+/- 524)
test bench_water3b              ... bench:       7,666 ns/iter (+/- 23)
test bench_water4               ... bench:     495,992 ns/iter (+/- 5,783)
test bench_water_huge           ... bench:  30,045,589 ns/iter (+/- 539,955)
test bench_water_huge2          ... bench:  60,316,824 ns/iter (+/- 2,158,995)
```

* Itrating through the vector of nodes instead of through the linked list indexes

As this 'simulates' a linked list, we can and do iterate through
using the next / previous pointers (which are indexes into a vector). 

However once in a while we need to do something that we can do better
by just iterating through the vector instead. If we need to touch
every node for example to translate the polygon to 0,0 for minx and miny,
we did that over the vector of nodes, instead of traversing the linked
list. 

As you may know, Vector in RAM is basically an array,and most CPU and 
memory circuitry is designed for doing fast work on arrays of data, for
various reasons. I suppose in theory you could design a linked list
computer that was faster at traversing linked lists, but most machines
really have optimization built in for arrays.

* Bounding Box

In which we compute the bounding box directly from the vector
holding the node information, instead of iterating
through the linked list.

During initial linked list setup we have to create a bounding box, find
the minimum x and y, and the maximum x and y, for all the points.
We can do this by iterating through the original data, or by iterating
through nodes.

rustfmt expands the first a huge amount taking up many lines of space,
becasue we have to skip(1) to hit the y data and clone() to deal with the
way the data is stored...or...

```rust
    let maxx = ll.nodes.iter().fold(std::f64::MIN, |m, n| f64::max(m, n.x));
    let maxy = ll.nodes.iter().fold(std::f64::MIN, |m, n| f64::max(m, n.y));
    let minx = ll.nodes.iter().fold(std::f64::MAX, |m, n| f64::min(m, n.x));
    let miny = ll.nodes.iter().fold(std::f64::MAX, |m, n| f64::min(m, n.y));
    let invsize = calc_invsize(minx, miny, maxx, maxy);
```

this is much shorter than C++, previous rust, javascript, etc, and
it runs in basically the same time.

again we are using the Vector to our advantage, instead of iterating
through as a linked list, iterate as a vector. CPUs and RAM circuitry
are designed for vectors, for various reasons.

```bash
test bench_water                ... bench:   1,965,894 ns/iter (+/- 47,469)
test bench_water2               ... bench:   1,493,086 ns/iter (+/- 13,284)
test bench_water3               ... bench:      67,764 ns/iter (+/- 398)
test bench_water3b              ... bench:       7,581 ns/iter (+/- 27)
test bench_water4               ... bench:     492,737 ns/iter (+/- 7,801)
test bench_water_huge           ... bench:  29,906,773 ns/iter (+/- 525,834)
test bench_water_huge2          ... bench:  60,744,602 ns/iter (+/- 2,544,614)
```

* Vector::with_capacity

In which we pre-compute vector capacity, saving a few 'resize'/reallocate
operations. 

We store two big chunks of data in Vectors, the nodes and the resulting
triangles. 

The nodes, well, they are nodes. They have point coordinates, and
links to other nodes, and the z-order hash value.

The triangles? Well, they are just indexes into the original data.

But we construct both of these vectors at runtime. We have to 
grow the vectors as the data is added to them, first during construction
of the nodes, then as the nodes are cut off by the earcutter, we have
to grow the triangles vector.

Or do we? Rust Vectors, like most vectors in computer lamnguages, have
'capacity'. You can reserve a big chunk of RAM for your vector even 
if you dont use it all right away. Then, subsequent 'push' operations
are really just filing in RAM you already had reserved for yourself. 

We can do this with our data... ok. I did that a while back, forgot to
even mention it.

But... we can also do it for our triangles. And we get a tiny little
boost..  stranegly enough on the smaller shapes :

```rust
running 7 tests
test bench_water                ... bench:   1,972,024 ns/iter (+/- 53,100)
test bench_water2               ... bench:   1,483,504 ns/iter (+/- 11,269)
test bench_water3               ... bench:      66,898 ns/iter (+/- 518)
test bench_water3b              ... bench:       7,276 ns/iter (+/- 6)
test bench_water4               ... bench:     489,625 ns/iter (+/- 2,844)
test bench_water_huge           ... bench:  29,874,170 ns/iter (+/- 494,521)
test bench_water_huge2          ... bench:  60,463,148 ns/iter (+/- 2,343,911)
```


* index_curve

In which we try to speedup the space filling curve code, and fail. 

tried to optimize with vector iteration, failed.

it iterates through a ring, z-indexing each point.

the werd thing is that as the algorithm overall progresses,
the number of 'unused' vector nodes shrinks. 

the issue is that if you use vector iteration, you wind up
iterating over a bunch of nodes that have been removed from the 
polygon. cut already into ears. so overall its slower.

even if you 'mark' removed nodes, it winds up being slower.

not much slower. and only shows on the huge shapes but still. not worth it.


* find_hole_bridge into iterator... uhm wow

In which we replace another for loop with iterators + functional code.

This is a very complicated function that does something simple in idea,
find a line between the left most point of a hole and the outer polygon
to connect it to.

The original code is something like this:

```rust
    loop {
        let (px, py) = (noderef!(ll, p).x, noderef!(ll, p).y);
        if (hy <= py) && (hy >= next!(ll, p).y) && (next!(ll, p).y != py) {
            let x = px + (hy - py) * (next!(ll, p).x - px) / (next!(ll, p).y - p
y);

            if (x <= hx) && (x > qx) {
                qx = x;
```

Into an iterator, after a few hours experimentation and work with iterators:

```rust
    let calcx =|p:&Node| p.x + (hy - p.y) * (next!(ll,p.idx).x - p.x) / (next!($
    for p in
        ll.iter_range(p..outer_node)
        .filter(|p| hy <= p.y)
        .filter(|p| hy >= next!(ll, p.idx).y)
        .filter(|p| next!(ll, p.idx).y != p.y)
        .filter(|p| calcx(p) <= hx )
    {
```

the speedup on huge and huge2 is ... apparently significant.

```bash
test bench_water                ... bench:   1,941,360 ns/iter (+/- 87,985)
test bench_water2               ... bench:   1,469,714 ns/iter (+/- 9,538)
test bench_water3               ... bench:      68,554 ns/iter (+/- 534)
test bench_water3b              ... bench:       7,269 ns/iter (+/- 34)
test bench_water4               ... bench:     487,715 ns/iter (+/- 7,659)
test bench_water_huge           ... bench:  28,610,478 ns/iter (+/- 803,598)
test bench_water_huge2          ... bench:  57,362,865 ns/iter (+/- 2,190,260)
```


Pulling the same trick with the bottom of the same function... uhm.. yeah.



before:
```rust
    while p != stop {
        let (px, py) = (noderef!(ll, p).x, noderef!(ll, p).y);
        if (hx >= px) && (px >= mx) && (hx != px) && point_in_triangle(&n1, &mp, &n2, noderef!(ll, p))
        {
            tan = (hy - py).abs() / (hx - px); // tangential
            if ((tan < tan_min) || ((tan == tan_min) && (px > noderef!(ll, m).x)))
                && locally_inside(ll, noderef!(ll, p), &hole)
            {
                m = p;
                tan_min = tan;
            }
        }
        p = next!(ll, p).idx;
    }
	return m
```

after:
```rust
    let calctan = |p: &Node| (hy - p.y).abs() / (hx - p.x); // tangential
    ll.iter_range(p..m)
        .filter(|p| hx > p.x && p.x >= mp.x)
                .filter(|p| point_in_triangle(&n1, &mp, &n2, &p))
        .fold((m, std::f64::INFINITY), |(m, tan_min), p| {
            if ((calctan(p) < tan_min) || (calctan(p) == tan_min && p.x > noderef!(ll, m).x))
                && locally_inside(ll, &p, noderef!(ll, hole))
            {
                (p.idx, calctan(p))
            } else {
                (m, tan_min)
            }
        })
        .0

```

.... errr well.


```bash
running 7 tests
test bench_water                ... bench:   1,886,095 ns/iter (+/- 48,910)
test bench_water2               ... bench:   1,455,139 ns/iter (+/- 8,578)
test bench_water3               ... bench:      65,432 ns/iter (+/- 851)
test bench_water3b              ... bench:       7,236 ns/iter (+/- 30)
test bench_water4               ... bench:     504,135 ns/iter (+/- 29,033)
test bench_water_huge           ... bench:  26,974,404 ns/iter (+/- 569,252)
test bench_water_huge2          ... bench:  53,630,310 ns/iter (+/- 1,818,475)
```

we are now within 50,000 ns of optimized C++ for bench_water or about 
.05 ms, 5%. 600,000 ns (0.6ms) for water_huge and uhm. we beat C++ water 
water_huge2 

For the full, final benchmark comparison to C++, see README.MD

* Thanks

Thanks for reading. 

Inspired by Michael Abrash's book from the 1990s about optimization.
