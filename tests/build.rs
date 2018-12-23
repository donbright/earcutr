/*
	build.rs -> part of the pre-test process, using cargo's 'build script' 
	system to pre-create tests by writing to a .rs file, which is later
	included by the main test .rs file.
*/
use std::fs::OpenOptions;
use std::io::Write;
fn main() {
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
