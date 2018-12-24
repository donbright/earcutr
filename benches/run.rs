#[macro_use]
extern crate bencher;
extern crate earcutr;

use bencher::Bencher;

fn basic_quadrilateral(bench: &mut Bencher) {
    bench.iter(|| {
		for _ in 0..9999 {
		    let triangles = earcutr::earcut(&vec![10., 0., 0., 50., 60., 60., 70., 10.],&vec![],2);
		    assert!(triangles==vec![1, 0, 3, 3, 2, 1]);
		}
    })
}

fn basic_hole(bench: &mut Bencher) {
    bench.iter(|| {
		for _ in 0..9999 {
			let mut v = vec![0., 0., 50., 0., 50., 50., 0., 50.];
			let h = vec![10., 10., 40., 10., 40., 40., 10., 40. ];
			v.extend(h);
		    let triangles = earcutr::earcut(&v,&vec![4],2);
		    assert!(triangles==vec![3,0,4,5,4,0,3,4,7,5,0,1,2,3,7,6,5,1,2,7,6,6,1,2]);
		}
    })
} 

benchmark_group!(benches, basic_quadrilateral, basic_hole);
benchmark_main!(benches);

