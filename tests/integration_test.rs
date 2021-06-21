extern crate earcutr;

extern crate serde;
extern crate serde_json;

use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;

static DEBUG: usize = 1;
macro_rules! dlog {
    ($loglevel:expr, $($s:expr),*) => (
        if DEBUG>=$loglevel { print!("{}:",$loglevel); println!($($s),+); }
    )
}

//fn format_percent(num: f64) -> String {
//    return num.to_string();
//String::from("1.234");
//    return ((1e8 * num).round() / 1e6).to_string();// + "%";
//}

fn parse_json(rawdata: &str) -> Option<Vec<Vec<Vec<f64>>>> {
    let mut v: Vec<Vec<Vec<f64>>> = Vec::new();
    match serde_json::from_str::<serde_json::Value>(&rawdata) {
        Err(e) => println!("error deserializing, {}", e),
        Ok(jsondata) => {
            if jsondata.is_array() {
                let contours = jsondata.as_array().unwrap();
                dlog!(4, "deserialize ok, {} contours", contours.len());
                for i in 0..contours.len() {
                    let contourval = &contours[i];
                    if contourval.is_array() {
                        let contour = contourval.as_array().unwrap();
                        dlog!(9, "countour {} numpoints {}", i, contour.len());
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

fn mkoutput(
    filename_w_dashes: &str,
    tris: Vec<usize>,
    data: &Vec<Vec<Vec<f64>>>,
    pass: bool,
    rpt: &str,
) -> Result<(), std::io::Error> {
    dlog!(
        4,
        "save data + triangles: {}, num tri pts:{}, rpt: {},",
        &filename_w_dashes,
        tris.len(),
        rpt
    );
    let filename = str::replace(filename_w_dashes, "-", "_");
    let outfile = &format!("viz/testoutput/{}.js", filename);
    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(outfile)
        .unwrap();
    writeln!(&f, r###"testOutput["{}"]=[];"###, filename)?;
    writeln!(&f, r###"testOutput["{}"]["json"]={:?};"###, filename, data)?;
    writeln!(
        &f,
        r###"testOutput["{}"]["triangles"]={:?};"###,
        filename, tris
    )?;
    writeln!(&f, r###"testOutput["{}"]["pass"]={:?};"###, filename, pass)?;
    writeln!(&f, r###"testOutput["{}"]["report"]={:?};"###, filename, rpt)?;
    dlog!(4, "wrote results to {}", outfile);
    Ok(())
}

// verify if triangles cover the same area as the shape itself
fn area_test(filename: &str, expected_num_tris: usize, expected_deviation: f64) -> bool {
    //    let visualize = std::env::args().any(|x| x == "--test-threads=1");
    let visualize = true;
    dlog!(4, "visualization: {}", visualize);
    let mut actual_num_tris = 0;
    let mut actual_deviation = 0.0;
    let mut edeviation = expected_deviation;
    let mut triangles: Vec<usize> = Vec::new();
    let mut xdata: Vec<Vec<Vec<f64>>> = Vec::new();
    if edeviation == 0.0 {
        edeviation = 1e-14;
    }
    let fullname = format!("tests/fixtures/{}.json", filename);
    match File::open(&fullname) {
        Err(why) => panic!("failed to open file '{}': {}", fullname, why),
        Ok(mut f) => {
            dlog!(4, "testing {},", fullname);
            let mut strdata = String::new();
            match f.read_to_string(&mut strdata) {
                Err(why) => dlog!(4, "failed to read {}, {}", fullname, why),
                Ok(numb) => {
                    dlog!(4, "read {} bytes", numb);
                    let rawstring = strdata.trim();
                    match parse_json(rawstring) {
                        None => dlog!(4, "failed to parse {}", fullname),
                        Some(parsed_data) => {
                            xdata = parsed_data;
                            let (data, holeidxs, dimensions) = earcutr::flatten(&xdata);
                            triangles = earcutr::earcut(&data, &holeidxs, dimensions);
                            actual_num_tris = triangles.len() / 3;
                            actual_deviation =
                                earcutr::deviation(&data, &holeidxs, dimensions, &triangles);
                        }
                    };
                }
            };
        }
    };
    let mut pass = true;
    if expected_num_tris > 0 && (expected_num_tris < actual_num_tris) {
        pass = false;
    };
    if edeviation < actual_deviation {
        pass = false;
    };
    let rpt = format!(
        "exp numtri:{}\nexp dev:{}\nact numtri:{}\nact dev:{}",
        expected_num_tris, edeviation, actual_num_tris, actual_deviation
    );
    if visualize {
        match mkoutput(&filename, triangles, &xdata, pass, &rpt) {
            Err(e) => println!("error writing output {}", e),
            _ => {}
        }
    }
    pass
}

// inline data based tests

#[test]
fn test_indices_2d() {
    let indices = earcutr::earcut(
        &vec![10.0, 0.0, 0.0, 50.0, 60.0, 60.0, 70.0, 10.0],
        &vec![],
        2,
    );
    assert!(indices == vec![1, 0, 3, 3, 2, 1]);
}

/*
#[test]
fn test_indices_3d() {
    let indices = earcutr::earcut(
        &vec![
            10.0, 0.0, 0.0, 0.0, 50.0, 0.0, 60.0, 60.0, 0.0, 70.0, 10.0, 0.0,
        ],
        &vec![],
        3,
    );
    assert!(indices == vec![1, 0, 3, 3, 2, 1]);
}
*/

#[test]
fn test_empty() {
    let indices = earcutr::earcut::<f64>(&vec![], &vec![], 2);
    println!("{:?}", indices);
    assert!(indices.len() == 0);
}

// file based tests

#[test]
fn test_building() {
    assert!(area_test("building", 13, 0e0));
}

#[test]
fn test_dude() {
    assert!(area_test("dude", 106, 0e0));
}

#[test]
fn test_water() {
    assert!(area_test("water", 2482, 8e-4));
}

#[test]
fn test_water2() {
    assert!(area_test("water2", 1212, 0e0));
}

#[test]
fn test_water3() {
    assert!(area_test("water3", 197, 0e0));
}

#[test]
fn test_water3b() {
    assert!(area_test("water3b", 25, 0e0));
}

#[test]
fn test_water4() {
    assert!(area_test("water4", 705, 0e0));
}

#[test]
fn test_water_huge() {
    assert!(area_test("water-huge", 5174, 1.1e-3));
}

#[test]
fn test_water_huge2() {
    assert!(area_test("water-huge2", 4461, 2.8e-3));
}

#[test]
fn test_degenerate() {
    assert!(area_test("degenerate", 0, 0e0));
}

#[test]
fn test_bad_hole() {
    assert!(area_test("bad-hole", 42, 1.9e-2));
}

#[test]
fn test_empty_square() {
    assert!(area_test("empty-square", 0, 0e0));
}

#[test]
fn test_issue16() {
    assert!(area_test("issue16", 12, 0e0));
}

#[test]
fn test_issue17() {
    assert!(area_test("issue17", 11, 0e0));
}

#[test]
fn test_steiner() {
    assert!(area_test("steiner", 9, 0e0));
}

#[test]
fn test_issue29() {
    assert!(area_test("issue29", 40, 0e0));
}

#[test]
fn test_issue34() {
    assert!(area_test("issue34", 139, 0e0));
}

#[test]
fn test_issue35() {
    assert!(area_test("issue35", 844, 0e0));
}

#[test]
fn test_self_touching() {
    assert!(area_test("self-touching", 124, 3.4e-14));
}

#[test]
fn test_outside_ring() {
    assert!(area_test("outside-ring", 64, 0e0));
}

#[test]
fn test_simplified_us_border() {
    assert!(area_test("simplified-us-border", 120, 0e0));
}

#[test]
fn test_touching_holes() {
    assert!(area_test("touching-holes", 57, 0e0));
}

#[test]
fn test_hole_touching_outer() {
    assert!(area_test("hole-touching-outer", 77, 0e0));
}

#[test]
fn test_hilbert() {
    assert!(area_test("hilbert", 1024, 0e0));
}

#[test]
fn test_issue45() {
    assert!(area_test("issue45", 10, 0e0));
}

#[test]
fn test_eberly_3() {
    assert!(area_test("eberly-3", 73, 0e0));
}

#[test]
fn test_eberly_6() {
    assert!(area_test("eberly-6", 1429, 0e0));
}

#[test]
fn test_issue52() {
    assert!(area_test("issue52", 109, 0e0));
}

#[test]
fn test_shared_points() {
    assert!(area_test("shared-points", 4, 0e0));
}

#[test]
fn test_bad_diagonals() {
    assert!(area_test("bad-diagonals", 7, 0e0));
}

#[test]
fn test_issue83() {
    assert!(area_test("issue83", 0, 1e-14));
}
