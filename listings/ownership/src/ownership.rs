fn main() {
    let s1 = String::from("foo");
    let s2 = s1;
    // println!("{s1}");

    let s1;
    {
        let s2 = String::from("bar");
        s1 = &s2;
        println!("{s1}");
    }
    println!("{s1}");
}
