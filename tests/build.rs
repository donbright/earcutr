/*
	build.rs -> part of the pre-test process, using cargo's 'build script' 
	system to pre-create tests by writing to a .rs file, which is later
	included by the main test .rs file.
*/
use std::fs::OpenOptions;
use std::io::Write;

fn test_list() -> Vec<(&'static str, usize, f32)> {
    vec![
        ("building", 13, 0.0),
        ("dude", 106, 0.0),
        ("water", 2482, 0.0008),
        ("water2", 1212, 0.0),
        ("water3", 197, 0.0),
        ("water3b", 25, 0.0),
        ("water4", 705, 0.0),
        ("water-huge", 5174, 0.0011),
        ("water-huge2", 4461, 0.0028),
        ("degenerate", 0, 0.0),
        ("bad-hole", 42, 0.019),
        ("empty-square", 0, 0.0),
        ("issue16", 12, 0.0),
        ("issue17", 11, 0.0),
        ("steiner", 9, 0.0),
        ("issue29", 40, 0.0),
        ("issue34", 139, 0.0),
        ("issue35", 844, 0.0),
        ("self-touching", 124, 3.4e-14),
        ("outside-ring", 64, 0.0),
        ("simplified-us-border", 120, 0.0),
        ("touching-holes", 57, 0.0),
        ("hole-touching-outer", 77, 0.0),
        ("hilbert", 1024, 0.0),
        ("issue45", 10, 0.0),
        ("eberly-3", 73, 0.0),
        ("eberly-6", 1429, 0.0),
        ("issue52", 109, 0.0),
        ("shared-points", 4, 0.0),
        ("bad-diagonals", 7, 0.0),
        ("issue83", 0, 1e-14),
        ("simple-touch", 2, 0.0),
    ]
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let destination = std::path::Path::new(&out_dir).join("test.rs");
    let mut f = std::fs::File::create(&destination).unwrap();
    println!("{:?}", destination);

    let txt = r###"
#[test]
fn __testname__() -> Result<(), String> {
    return area_test("__filename__",__numtris__,__area__);
}
"###;

    for test in test_list().iter() {
        let (filename, expect_numtris, expect_area) = (test.0, test.1, test.2);
        let mut newtxt = txt.replace("__testname__", &filename.replace("-", "_"));
        newtxt = newtxt.replace("__filename__", filename);
        newtxt = newtxt.replace("__numtris__", &format!("{}", expect_numtris));
        newtxt = newtxt.replace("__area__", &format!("{:e}", expect_area));
        f.write(newtxt.as_bytes()).unwrap();
    }

    // this filename + variablename also in integration_test.rs and viz.html
    let ofile = "viz/testoutput.js";
    match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(ofile)
    {
        Err(e) => println!("failed to create {}, {}", ofile, e),
        Ok(mut f) => writeln!(f, "testOutput=[];").unwrap(),
    };
}
