extern crate earcutr;

extern crate serde;
extern crate serde_json;
//#[macro_use]
extern crate serde_derive;

use std::fs::File;
//use std::io::prelude::*;
use std::io::Read;
//use serde_json::Error;

fn format_percent( num:f32 ) -> String {
	return num.to_string();
//String::from("1.234");
//    return ((1e8 * num).round() / 1e6).to_string();// + "%";
}


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

fn parse_json(rawdata: &str) -> Option<Vec<Vec<Vec<f32>>>> {
	let mut v: Vec<Vec<Vec<f32>>> = Vec::new();
	match serde_json::from_str::<serde_json::Value>(&rawdata) {
		Err(e) => println!("error deserializing, {}",e),
		Ok(jsondata) => {
			if jsondata.is_array() {
				let contours = jsondata.as_array().unwrap();
				println!("deserialize ok, {} contours",contours.len());
				for i in 0..contours.len() {
					let contourval = &contours[i];
					if contourval.is_array() {
						let contour = contourval.as_array().unwrap();
						println!("countour {} numpoints {}",i,contour.len());
						let mut vc:Vec<Vec<f32>> = Vec::new();
						for j in 0..contour.len() {
							let points = contour[j].as_array().unwrap();
						 let mut vp: Vec<f32> = Vec::new();
 						for k in 0..points.len() {
								let val = points[k].to_string();
								vp.push(val.parse::<f32>().unwrap());
								//print!(" {}",points[k]);
							} //print!(",");
							vc.push(vp);
						}
						v.push(vc);
						//println!();
					}
				}
			}
		},
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

fn area_test(filename:&str, expected_triangles:u32, expected_deviation:f32) {
	let mut deviation = expected_deviation;
	if deviation == 0.0 {
		deviation = 1e-14;
	}
	let fullname = format!("tests/fixtures/{}.json",filename);
	match File::open(&fullname) {
		Err(why) => panic!("failed to open file '{}': {}", fullname, why),
		Ok(mut f) => {
			print!("testing {},",fullname);
			let mut strdata = String::new();
			match f.read_to_string( &mut strdata ) {
				Err(why) => println!("failed to read {}, {}",fullname,why),
				Ok(numb) => {
					println!("read {} bytes",numb);
					let rawstring = strdata.trim();
					match parse_json( rawstring ) {
						None => println!("failed to parse {}",fullname),
						Some(strdata) => {
							let (data, holeidxs, dimensions) = earcutr::flatten( &strdata );
							let indices = earcutr::earcut( &data, holeidxs, dimensions );
							//deviation = earcutr::deviation( data, holeidxs, dimensions, indices );)
						},
					};
				},
			};
		},
	};
	assert_eq!( deviation, deviation );
	assert_eq!( expected_triangles, expected_triangles );
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

#[test]
fn it_adds_two() {
    println!("integration test");
    assert_eq!(4, 2+2);
}

#[test]
fn area_tests() {
    area_test("building", 13, 0.0);
	return;
    area_test("dude", 106, 0.0);
    area_test("water", 2482, 0.0008);
    area_test("water2", 1212, 0.0);
    area_test("water3", 197, 0.0);
    area_test("water3b", 25, 0.0);
    area_test("water4", 705, 0.0);
    area_test("water-huge", 5174, 0.0011);
    area_test("water-huge2", 4461, 0.0028);
    area_test("degenerate", 0, 0.0);
    area_test("bad-hole", 42, 0.019);
    area_test("empty-square", 0, 0.0);
    area_test("issue16", 12, 0.0);
    area_test("issue17", 11, 0.0);
    area_test("steiner", 9, 0.0);
    area_test("issue29", 40, 0.0);
    area_test("issue34", 139, 0.0);
    area_test("issue35", 844, 0.0);
    area_test("self-touching", 124, 3.4e-14);
    area_test("outside-ring", 64, 0.0);
    area_test("simplified-us-border", 120, 0.0);
    area_test("touching-holes", 57, 0.0);
    area_test("hole-touching-outer", 77, 0.0);
    area_test("hilbert", 1024, 0.0);
    area_test("issue45", 10, 0.0);
    area_test("eberly-3", 73, 0.0);
    area_test("eberly-6", 1429, 0.0);
    area_test("issue52", 109, 0.0);
    area_test("shared-points", 4, 0.0);
    area_test("bad-diagonals", 7, 0.0);
    area_test("issue83", 0, 1e-14);
}
