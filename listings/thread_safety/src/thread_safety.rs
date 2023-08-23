const NELEMENTS: usize = 1_000_000;
const NTHREADS: usize = 8;
const QOT: usize = NELEMENTS / NTHREADS;
const REM: usize = NELEMENTS % NTHREADS;

fn main() {
    let array = vec![1; NELEMENTS];
    let mut threads = Vec::with_capacity(NTHREADS);

    let mut result = 0;
    for t in 0..NTHREADS {
        let start = t * QOT;
        let end = if t == NTHREADS - 1 {
            start + QOT + REM
        } else {
            start + QOT
        };
        threads.push(std::thread::spawn(|| {
            for i in start..end {
                result += array[i];
            }
        }));
    }

    for t in threads {
        t.join().unwrap();
    }

    println!("RESULT: {result}");
}
