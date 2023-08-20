const NELEMENTS: usize = 1_000_000;
const NTHREADS: usize = 4;
const QOT: usize = NELEMENTS / NTHREADS;
const REM: usize = NELEMENTS % NTHREADS;

fn main() {
    let array = vec![1.0; NELEMENTS];
    let mut threads = Vec::with_capacity(NTHREADS);
    let mut start = 0;

    let mut result = 0.0;
    for t in 0..NTHREADS {
        let loc = if t < REM { QOT + 1 } else { QOT };
        let end = start + loc;
        threads.push(std::thread::spawn(|| {
            let mut partial_sum = 0.0;
            for i in start..end {
                partial_sum += array[i];
            }
            result += partial_sum;
        }));
        start = end;
    }

    for t in threads {
        t.join().unwrap();
    }

    println!("result: {result}");
}
