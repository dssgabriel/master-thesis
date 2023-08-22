fn main() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 5.0];
    let sum: f64 = x.iter().sum();
    println!(" x = {x:?}");
    println!("Σx = {sum}");
}
