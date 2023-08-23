#show raw.where(block: true): it => {
    set text(font: "IBM Plex Mono")
    set align(left)
    set block(fill: luma(240), inset: 10pt, radius: 4pt, width: 100%)
    it
}

= Appendix

#figure(caption: "Rust's compiler error message for a race condition bug in")[
  ```
error[E0373]: closure may outlive the current function, but it borrows `result`, which is owned by
the current function
  --> src/thread_safety.rs:18:41
   |
18 |         threads.push(std::thread::spawn(|| {
   |                                         ^^ may outlive borrowed value `result`
19 |             for i in start..end {
20 |                 result += array[i];
   |                 ------ `result` is borrowed here
   |
note: function requires argument type to outlive `'static`
  --> src/thread_safety.rs:18:22
   |
18 |           threads.push(std::thread::spawn(|| {
   |  ______________________^
19 | |             for i in start..end {
20 | |                 result += array[i];
21 | |             }
22 | |         }));
   | |__________^
help: to force the closure to take ownership of `result` (and any other referenced variables), use
the `move` keyword
   |
18 |         threads.push(std::thread::spawn(move || {
   |                                         ++++

error[E0499]: cannot borrow `result` as mutable more than once at a time
  --> src/thread_safety.rs:18:41
   |
18 |           threads.push(std::thread::spawn(|| {
   |                        -                  ^^ `result` was mutably borrowed here in the previous iteration of the loop
   |  ______________________|
   | |
19 | |             for i in start..end {
20 | |                 result += array[i];
   | |                 ------ borrows occur due to use of `result` in closure
21 | |             }
22 | |         }));
   | |__________- argument requires that `result` is borrowed for `'static`
  ```
]<error_race_cond>