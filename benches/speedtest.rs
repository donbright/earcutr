#[macro_use]
extern crate bencher;
extern crate earcutr;

use bencher::Bencher;

fn quadrilateral(bench: &mut Bencher) {
    bench.iter(|| {
        for _ in 0..99 {
            let triangles =
                earcutr::earcut(&vec![10., 0., 0., 50., 60., 60., 70., 10.], &vec![], 2);
            assert!(triangles == vec![1, 0, 3, 3, 2, 1]);
        }
    })
}

fn hole(bench: &mut Bencher) {
    bench.iter(|| {
        for _ in 0..99 {
            let mut v = vec![0., 0., 50., 0., 50., 50., 0., 50.];
            let h = vec![10., 10., 40., 10., 40., 40., 10., 40.];
            v.extend(h);
            let triangles = earcutr::earcut(&v, &vec![4], 2);
            //let triangles= vec![3,0,4,5,4,0,3,4,7,5,0,1,2,3,7,6,5,1,2,7,6,6,1,2];
            assert!(
                triangles
                    == vec![3, 0, 4, 5, 4, 0, 3, 4, 7, 5, 0, 1, 2, 3, 7, 6, 5, 1, 2, 7, 6, 6, 1, 2]
            );
        }
    })
}

fn flatten(bench: &mut Bencher) {
    bench.iter(|| {
        for _ in 0..99 {
            let v = vec![
                vec![vec![0., 0.], vec![1., 0.], vec![1., 1.], vec![0., 1.]], // outer ring
                vec![vec![1., 1.], vec![3., 1.], vec![3., 3.]],               // hole ring
            ];
            let (vertices, holes, dimensions) = earcutr::flatten(&v);
            let triangles = earcutr::earcut(&vertices, &holes, dimensions);
            assert!(triangles.len() == 9);
        }
    })
}

fn badhole(bench: &mut Bencher) {
    bench.iter(|| {
        for _ in 0..99 {
            let v = vec![
                vec![
                    vec![810., 2828.],
                    vec![818., 2828.],
                    vec![832., 2818.],
                    vec![844., 2806.],
                    vec![855., 2808.],
                    vec![866., 2816.],
                    vec![867., 2824.],
                    vec![876., 2827.],
                    vec![883., 2834.],
                    vec![875., 2834.],
                    vec![867., 2840.],
                    vec![878., 2838.],
                    vec![889., 2844.],
                    vec![880., 2847.],
                    vec![870., 2847.],
                    vec![860., 2864.],
                    vec![852., 2879.],
                    vec![847., 2867.],
                    vec![810., 2828.],
                    vec![810., 2828.],
                ],
                vec![
                    vec![818., 2834.],
                    vec![823., 2833.],
                    vec![831., 2828.],
                    vec![839., 2829.],
                    vec![839., 2837.],
                    vec![851., 2845.],
                    vec![847., 2835.],
                    vec![846., 2827.],
                    vec![847., 2827.],
                    vec![837., 2827.],
                    vec![840., 2815.],
                    vec![835., 2823.],
                    vec![818., 2834.],
                    vec![818., 2834.],
                ],
                vec![
                    vec![857., 2846.],
                    vec![864., 2850.],
                    vec![866., 2839.],
                    vec![857., 2846.],
                    vec![857., 2846.],
                ],
                vec![
                    vec![848., 2863.],
                    vec![848., 2866.],
                    vec![854., 2852.],
                    vec![846., 2854.],
                    vec![847., 2862.],
                    vec![838., 2851.],
                    vec![838., 2859.],
                    vec![848., 2863.],
                    vec![848., 2863.],
                ],
            ];

            let (vertices, holes, dimensions) = earcutr::flatten(&v);
            let triangles = earcutr::earcut(&vertices, &holes, dimensions);
            assert!(triangles.len() == 126);
        }
    })
}

benchmark_group!(benches, quadrilateral, hole, flatten, badhole);
benchmark_main!(benches);
