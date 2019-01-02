/*
Note: This module uses the 'unofficial' bench that works with Stable Rust as
of 2018. This may conflict with "official" bench which is in "Nightly" Rust
*/
#[macro_use]
extern crate bencher;
extern crate earcutr;
extern crate serde;
extern crate serde_json;
use bencher::Bencher;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;

// this is to "force" optimized code to measure results, by outputting
fn mkoutput(filename_w_dashes: &str, triangles: Vec<usize>) {
    let filename = str::replace(filename_w_dashes, "-", "_");
    let outfile = &format!("benches/benchoutput/{}.js", filename);
    match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(outfile)
    {
        Err(e) => println!("error writing {} {}", outfile, e),
        Ok(f) => writeln!(
            &f,
            r###"testOutput["{}"]["benchmark"]=[{:?},{:?},{:?}];"###,
            filename,
            0,
            triangles.len(),
            triangles
        ).unwrap(),
    };
}

fn parse_json(rawdata: &str) -> Option<Vec<Vec<Vec<f64>>>> {
    let mut v: Vec<Vec<Vec<f64>>> = Vec::new();
    match serde_json::from_str::<serde_json::Value>(&rawdata) {
        Err(e) => println!("error deserializing, {}", e),
        Ok(jsondata) => {
            if jsondata.is_array() {
                let contours = jsondata.as_array().unwrap();
                for i in 0..contours.len() {
                    let contourval = &contours[i];
                    if contourval.is_array() {
                        let contour = contourval.as_array().unwrap();
                        let mut vc: Vec<Vec<f64>> = Vec::new();
                        for j in 0..contour.len() {
                            let points = contour[j].as_array().unwrap();
                            let mut vp: Vec<f64> = Vec::new();
                            for k in 0..points.len() {
                                let val = points[k].to_string();
                                let pval = val.parse::<f64>().unwrap();
                                vp.push(pval);
                            }
                            vc.push(vp);
                        }
                        v.push(vc);
                    }
                }
            }
        }
    };
    return Some(v);
}

fn load_json(testname: &str) -> (Vec<f64>, Vec<usize>, usize) {
    let fullname = format!("./tests/fixtures/{}.json", testname);
    let mut xdata: Vec<Vec<Vec<f64>>> = Vec::new();
    match File::open(&fullname) {
        Err(why) => println!("failed to open file '{}': {}", fullname, why),
        Ok(mut f) => {
            //println!("testing {},", fullname);
            let mut strdata = String::new();
            match f.read_to_string(&mut strdata) {
                Err(why) => println!("failed to read {}, {}", fullname, why),
                Ok(_numb) => {
                    //println!("read {} bytes", numb);
                    let rawstring = strdata.trim();
                    match parse_json(rawstring) {
                        None => println!("failed to parse {}", fullname),
                        Some(parsed_data) => {
                            xdata = parsed_data;
                        }
                    };
                }
            };
        }
    };
    return earcutr::flatten(&xdata);
}

fn bench_quadrilateral(bench: &mut Bencher) {
    bench.iter(|| {
        earcutr::earcut(&vec![10., 0., 0., 50., 60., 60., 70., 10.], &vec![], 2);
    });
}

fn bench_hole(bench: &mut Bencher) {
    let mut v = vec![0., 0., 50., 0., 50., 50., 0., 50.];
    let h = vec![10., 10., 40., 10., 40., 40., 10., 40.];
    v.extend(h);
    bench.iter(|| {
        earcutr::earcut(&v, &vec![4], 2);
    })
}

fn bench_flatten(bench: &mut Bencher) {
    let v = vec![
        vec![vec![0., 0.], vec![1., 0.], vec![1., 1.], vec![0., 1.]], // outer ring
        vec![vec![1., 1.], vec![3., 1.], vec![3., 3.]],               // hole ring
    ];
    bench.iter(|| {
        let (_vertices, _holes, _dimensions) = earcutr::flatten(&v);
    })
}

fn bench_indices_2d(bench: &mut Bencher) {
    bench.iter(|| {
        let _indices = earcutr::earcut(
            &vec![10.0, 0.0, 0.0, 50.0, 60.0, 60.0, 70.0, 10.0],
            &vec![],
            2,
        );
    })
}

fn bench_indices_3d(bench: &mut Bencher) {
    bench.iter(|| {
        let _indices = earcutr::earcut(
            &vec![
                10.0, 0.0, 0.0, 0.0, 50.0, 0.0, 60.0, 60.0, 0.0, 70.0, 10.0, 0.0,
            ],
            &vec![],
            3,
        );
    })
}

fn bench_empty(bench: &mut Bencher) {
    bench.iter(|| {
        let _indices = earcutr::earcut(&vec![], &vec![], 2);
    })
}

// file based tests

fn bench_building(bench: &mut Bencher) {
    let nm = "building";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_dude(bench: &mut Bencher) {
    let nm = "dude";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_water(bench: &mut Bencher) {
    let nm = "water";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_water2(bench: &mut Bencher) {
    let nm = "water2";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_water3(bench: &mut Bencher) {
    let nm = "water3";

    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_water3b(bench: &mut Bencher) {
    let nm = "water3b";

    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_water4(bench: &mut Bencher) {
    let nm = "water4";

    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_water_huge(bench: &mut Bencher) {
    let nm = "water-huge";

    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_water_huge2(bench: &mut Bencher) {
    let nm = "water-huge2";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_degenerate(bench: &mut Bencher) {
    let nm = "degenerate";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_bad_hole(bench: &mut Bencher) {
    let nm = "bad-hole";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_empty_square(bench: &mut Bencher) {
    let nm = "empty-square";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_issue16(bench: &mut Bencher) {
    let nm = "issue16";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_issue17(bench: &mut Bencher) {
    let nm = "issue17";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_steiner(bench: &mut Bencher) {
    let nm = "steiner";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_issue29(bench: &mut Bencher) {
    let nm = "issue29";
    let (data, holeidxs, dimensions) = load_json(nm);

    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_issue34(bench: &mut Bencher) {
    let nm = "issue34";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_issue35(bench: &mut Bencher) {
    let nm = "issue35";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_self_touching(bench: &mut Bencher) {
    let nm = "self-touching";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_outside_ring(bench: &mut Bencher) {
    let nm = "outside-ring";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_simplified_us_border(bench: &mut Bencher) {
    let nm = "simplified-us-border";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_touching_holes(bench: &mut Bencher) {
    let nm = "touching-holes";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_hole_touching_outer(bench: &mut Bencher) {
    let nm = "hole-touching-outer";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_hilbert(bench: &mut Bencher) {
    let nm = "hilbert";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_issue45(bench: &mut Bencher) {
    let nm = "issue45";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_eberly_3(bench: &mut Bencher) {
    let nm = "eberly-3";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_eberly_6(bench: &mut Bencher) {
    let nm = "eberly-6";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_issue52(bench: &mut Bencher) {
    let nm = "issue52";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_shared_points(bench: &mut Bencher) {
    let nm = "shared-points";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_bad_diagonals(bench: &mut Bencher) {
    let nm = "bad-diagonals";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

fn bench_issue83(bench: &mut Bencher) {
    let nm = "issue83";
    let (data, holeidxs, dimensions) = load_json(nm);
    let mut triangles = Vec::new();
    bench.iter(|| {
        triangles = earcutr::earcut(&data, &holeidxs, dimensions);
    });
    mkoutput(nm, triangles);
}

benchmark_group!(
    benches,
    bench_indices_3d,
    bench_indices_2d,
    bench_empty,
    bench_quadrilateral,
    bench_hole,
    bench_flatten,
    bench_bad_diagonals,
    bench_bad_hole,
    bench_building,
    bench_degenerate,
    bench_dude,
    bench_eberly_3,
    bench_eberly_6,
    bench_empty_square,
    bench_hilbert,
    bench_hole_touching_outer,
    bench_issue16,
    bench_issue17,
    bench_issue29,
    bench_issue34,
    bench_issue35,
    bench_issue45,
    bench_issue52,
    bench_issue83,
    bench_outside_ring,
    bench_self_touching,
    bench_shared_points,
    bench_simplified_us_border,
    bench_steiner,
    bench_touching_holes,
    bench_water_huge,
    bench_water_huge2,
    bench_water,
    bench_water2,
    bench_water3,
    bench_water3b,
    bench_water4,
);
benchmark_main!(benches);
