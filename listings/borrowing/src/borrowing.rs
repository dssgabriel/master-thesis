fn main() {
    let s2;
    {
        let s1 = String::from("bar");
        s2 = &s1; // Borrowing the value held by `s1`
        println!("{s1}"); // OK! `s2` only has borrowed `s1`, thus it is still valid
        println!("{s2}"); // OK! `s2` holds a reference to "bar" but does not own it
    }
    println!("{s2}"); // ERROR! `s2` held a reference that is not valid anymore
                      // because the owner `s1` went out of scope

    let mut s1 = String::from("Hello, ");
    {
        let s2 = &mut s1; // Mutably borrowing the value held by `s1`
                          // let s3 = &s1; // ERROR! cannot borrow because `s2` is a mutable borrow in scope and is used later to modify `s1`
        s2.push_str("world!"); // Modifying `s1` through `s2`
                               // println!("{s3}");
                               // `s2` falls out of scope and drops the mutable reference
    }
    let s3 = &s1; // OK! The mutable reference to does not exist anymore `s1`
    println!("{s3}"); // Prints "Hello, world!"
}
