extern crate earcutr;

extern crate serde;
extern crate serde_json;
//#[macro_use]
extern crate serde_derive;

use std::fs::File;
//use std::io::prelude::*;
use std::fs::OpenOptions;
//use std::io::prelude::*;
use std::io::Read;
use std::io::Write;
//use serde_json::Error;

//fn format_percent(num: earcutr::Coordinate) -> String {
//    return num.to_string();
    //String::from("1.234");
    //    return ((1e8 * num).round() / 1e6).to_string();// + "%";
//}

/*
test("indices-2d", function (t) {
    var indices = earcut([10, 0, 0, 50, 60, 60, 70, 10]);
    t.same(indices, [1, 0, 3, 3, 2, 1]);
    t.end();
});

test("indices-3d", function (t) {
    var indices = earcut([10, 0, 0, 0, 50, 0, 60, 60, 0, 70, 10, 0], null, 3);
    t.same(indices, [1, 0, 3, 3, 2, 1]);
    t.end();
});

test("empty", function (t) {
    t.same(earcut([]), []);
    t.end();
});
*/

fn parse_json(rawdata: &str) -> Option<Vec<Vec<Vec<earcutr::Coordinate>>>> {
    let mut v: Vec<Vec<Vec<earcutr::Coordinate>>> = Vec::new();
    match serde_json::from_str::<serde_json::Value>(&rawdata) {
        Err(e) => println!("error deserializing, {}", e),
        Ok(jsondata) => {
            if jsondata.is_array() {
                let contours = jsondata.as_array().unwrap();
                //println!("deserialize ok, {} contours", contours.len());
                for i in 0..contours.len() {
                    let contourval = &contours[i];
                    if contourval.is_array() {
                        let contour = contourval.as_array().unwrap();
                        //println!("countour {} numpoints {}", i, contour.len());
                        let mut vc: Vec<Vec<earcutr::Coordinate>> = Vec::new();
                        for j in 0..contour.len() {
                            let points = contour[j].as_array().unwrap();
                            let mut vp: Vec<earcutr::Coordinate> = Vec::new();
                            for k in 0..points.len() {
                                let val = points[k].to_string();
								let pval = val.parse::<earcutr::Coordinate>().unwrap();
                                vp.push(pval);
                            } //print!(",");
                            vc.push(vp);
                        }
                        v.push(vc);
                        //println!();
                    }
                }
            }
        }
    };
    /* for i in 0..v.len() {
		for j in 0..v[i].len() {
			for k in 0..v[i][j].len() {
				print!("{},",v[i][j][k]);
			}
		}
	}
*/
    return Some(v);
}

fn mkoutput(filename: &str, tris: &Vec<usize>, data: &Vec<Vec<Vec<earcutr::Coordinate>>>, pass: bool, rpt:&str ) {
    println!(
        "save data + triangles: {}, num tri pts:{}, rpt: {},",
        &filename,
        tris.len(),rpt
    );
    // this filename + variablename also in integration_test.rs and viz.html
    let outfile = "viz/testoutput.js";
    let f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(outfile)
        .unwrap();
    writeln!(&f, r###"testOutput["{}"]=[];"###,filename);
    writeln!(&f, r###"testOutput["{}"]["json"]={:?};"###, filename, data);
    writeln!(
        &f,
        r###"testOutput["{}"]["triangles"]={:?};"###,
        filename, tris
    );
    writeln!(&f, r###"testOutput["{}"]["pass"]={:?};"###, filename,pass);
    writeln!(&f, r###"testOutput["{}"]["report"]={:?};"###, filename,rpt);
    println!("wrote results to {}", outfile);
}

fn area_test(filename: &str, expected_num_tris: usize, expected_deviation: earcutr::Coordinate) {
    let visualize = std::env::args().any(|x| x == "--test-threads=1");
    println!("visualization: {}", visualize);
	let mut actual_num_tris = 0;
	let mut actual_deviation = 0.0;
    let mut edeviation = expected_deviation;
	let mut triangles:Vec<usize> = Vec::new();
	let mut xdata:Vec<Vec<Vec<earcutr::Coordinate>>> = Vec::new();
    if edeviation == 0.0 {
        edeviation = 1e-14;
    }
    let fullname = format!("tests/fixtures/{}.json", filename);
    match File::open(&fullname) {
        Err(why) => panic!("failed to open file '{}': {}", fullname, why),
        Ok(mut f) => {
            print!("testing {},", fullname);
            let mut strdata = String::new();
            match f.read_to_string(&mut strdata) {
                Err(why) => println!("failed to read {}, {}", fullname, why),
                Ok(numb) => {
                    println!("read {} bytes", numb);
                    let rawstring = strdata.trim();
                    match parse_json(rawstring) {
                        None => println!("failed to parse {}", fullname),
                        Some(parsed_data) => {
							xdata = parsed_data;
                            let (data, holeidxs, dimensions) = earcutr::flatten(&xdata);
                            triangles = earcutr::earcut(&data, &holeidxs, dimensions);
							actual_num_tris = triangles.len()/3;
                            actual_deviation = earcutr::deviation( &data, &holeidxs, dimensions, &triangles );
                        }
                    };
                }
            };
        }
    };
	let mut pass = true;
    if expected_num_tris>0 && (expected_num_tris !=  actual_num_tris) { pass = false; };
	if edeviation < actual_deviation { pass = false; };
    if visualize {
		let rpt = format!("exp numtri:{}\nexp dev:{}\nact numtri:{}\nact dev:{}",
			expected_num_tris,edeviation, actual_num_tris, actual_deviation);
        mkoutput(&filename, &triangles, &xdata, pass, &rpt);
    }
	assert_eq!( pass, true );
}

/*
function area_test(filename, expectedTriangles, expectedDeviation) {
    expectedDeviation = expectedDeviation || 1e-14;

    test(filename, function (t) {

        var data = earcut::flatten(JSON.parse(
fs.readFileSync(path.join(__dirname, "/fixtures/" + filename + ".json")
))),
            indices = earcut(data.vertices, data.holes, data.dimensions),
            deviation = earcut.deviation(data.vertices, data.holes, data.dimensions, indices);

        t.ok(deviation < expectedDeviation,
            "deviation " + formatPercent(deviation) + " is less than " + formatPercent(expectedDeviation));

        if (expectedTriangles) {
            var numTriangles = indices.length / 3;
            t.ok(numTriangles === expectedTriangles, numTriangles + " triangles when expected " + expectedTriangles);
        }

        t.end();
    });
}
*/

// #[test] lines are generated by 'build.rs' at compile time
// they are dumped into a file called test.rs under the build dir,included below
// each test calls area_test( filename, expected_tris, expected_area )
include!(concat!(env!("OUT_DIR"), "/test.rs"));
