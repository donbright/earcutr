extern crate earcutr;

extern crate serde;
extern crate serde_json;
extern crate serde_derive;

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

fn parse_json(rawdata: &str) -> Option<Vec<Vec<Vec<f64>>>> {
    let mut v: Vec<Vec<Vec<f64>>> = Vec::new();
    match serde_json::from_str::<serde_json::Value>(&rawdata) {
        Err(e) => println!("error deserializing, {}", e),
        Ok(jsondata) => {
            if jsondata.is_array() {
                let contours = jsondata.as_array().unwrap();
                dlog!(4,"deserialize ok, {} contours", contours.len());
                for i in 0..contours.len() {
                    let contourval = &contours[i];
                    if contourval.is_array() {
                        let contour = contourval.as_array().unwrap();
                        dlog!(9,"countour {} numpoints {}", i, contour.len());
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

fn mkoutput(filename: &str, tris: &Vec<usize>, data: &Vec<Vec<Vec<f64>>>, pass: bool, rpt:&str ) {
    dlog!(4,
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
    dlog!(4,"wrote results to {}", outfile);
}

// this is called by test.rs, which is generated at compile time by 
// build.rs running at the first stage of 'cargo test'.
fn area_test(filename: &str, expected_num_tris: usize, expected_deviation: f64) -> Result<(), String> {
    let visualize = std::env::args().any(|x| x == "--test-threads=1");
    dlog!(4,"visualization: {}", visualize);
	let mut actual_num_tris = 0;
	let mut actual_deviation = 0.0;
    let mut edeviation = expected_deviation;
	let mut triangles:Vec<usize> = Vec::new();
	let mut xdata:Vec<Vec<Vec<f64>>> = Vec::new();
    if edeviation == 0.0 {
        edeviation = 1e-14;
    }
    let fullname = format!("tests/fixtures/{}.json", filename);
    match File::open(&fullname) {
        Err(why) => panic!("failed to open file '{}': {}", fullname, why),
        Ok(mut f) => {
            dlog!(4,"testing {},", fullname);
            let mut strdata = String::new();
            match f.read_to_string(&mut strdata) {
                Err(why) => dlog!(4,"failed to read {}, {}", fullname, why),
                Ok(numb) => {
                    dlog!(4,"read {} bytes", numb);
                    let rawstring = strdata.trim();
                    match parse_json(rawstring) {
                        None => dlog!(4,"failed to parse {}", fullname),
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
	let rpt = format!("exp numtri:{}\nexp dev:{}\nact numtri:{}\nact dev:{}",
		expected_num_tris,edeviation, actual_num_tris, actual_deviation);
    if visualize {
        mkoutput(&filename, &triangles, &xdata, pass, &rpt);
    }
	if pass { return Ok(()); }
	return Err(String::from( format!("{} {}",filename,rpt) ));
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

// #[test] lines and functions  are generated by 'build.rs' at compile time
// they are dumped into a file called test.rs under the build dir,included below
// each test calls area_test( filename, expected_tris, expected_area )
include!(concat!(env!("OUT_DIR"), "/test.rs"));
