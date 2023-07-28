# Master's thesis - Rust and GPU programming

## About
This repository contains the source files for my end-of-studies internship report/master's thesis at [CEA](https://www.cea.fr/english) (French Commission for Alternative Energies and Atomic Energy), as part of my [M.Sc. in High Performance Computing and Simulation](https://chps.uvsq.fr/) at [Paris-Saclay University](https://www.universite-paris-saclay.fr/en).

This project is written in [Typst](https://typst.app/) and thus requires you have it installed on your machine (see [Typst's documentation](https://github.com/typst/typst#installation) on the installation procedure depending on your system).

## Building
To build the PDF, you can either use the provided `justfile` (if you have [just](https://github.com/casey/just) installed):
```sh
just build
```
or building it manually using Typst:
```sh
typst compile src/main.typ --root . <OUTPUT_PDF_NAME>
```

## `justfile` features
The `justfile` also provides some more functionnalities like automatically compiling and opening the PDF, generating a tarball, etc... To see all the available recipes, simply type:
```sh
just
```

## Licenses
Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/cea-hpc/HARP/blob/master/LICENSE-APACHE) or [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0));
- MIT License ([LICENSE-MIT](https://github.com/cea-hpc/HARP/blob/master/LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
at your option.  

The [SPDX](https://spdx.dev/) license identifier for this project is MIT OR Apache-2.0.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
