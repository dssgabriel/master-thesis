#grid(
  columns: (1fr, 1fr),
  align(top + center)[
    #image("../figures/ups_logo.svg", width: 45%)
  ],
  align(center)[
    #image("../figures/cea_logo.svg", width: 40%)
  ]
)

#align(
  center,
  text(24pt)[
    Paris-Saclay University
  ]
)
#align(
  center,
  text(18pt)[
    Master 2 High Performance Computing, Simulation \
    End-of-studies internship report\
  ]
)

#line(length: 100%)
#v(1.8em)
#align(center, text(32pt)[*Rust and GPU programming*])
#line(length: 100%)

#v(1em)
#grid(columns: (1fr, 1fr),
  align(center, box(align(start, text(16pt)[
      *Author :* \
      Gabriel #smallcaps("Dos Santos")
    ]))),
  align(center, box(align(end, text(16pt)[
      *Supervisors :* \
      Cédric #smallcaps("Chevalier") \
      Soraya #smallcaps("Zertal")
    ])))
)

#v(1fr)

#align(center, text(13pt)[*Abstract*])
#text(11pt)[
In recent years, the field of High Performance Computing (HPC) has known significant advances in hardware technology. With the advent of heterogeneous architectures, these improvements imply an ever-increasing need for hardware accelerator programming, which is essential to harness the computational performance of exascale supercomputers.
#linebreak()
To meet these new requirements, software engineering and programming languages are evolving to improve programmers' control and safety over parallel applications. Rust is a modern programming language focused on performance, memory safety, and concurrency. Its features make it an ideal choice for guaranteeing the robustness and efficiency of compute-intensive codes, particularly when parallelizing certain forms of data-flow algorithms.

In this context, the CEA uses Rust to develop several applications and tools, some of which could benefit from the accelerations offered by GPUs. This internship aims to explore the capabilities of the Rust language for programming hardware accelerators. This work consists of a detailed overview of the current state of the art, performance analysis depending on the chosen code generation method, and a proof of concept by porting partitioning algorithms from CEA's #link("https://github.com/LIHPC-Computational-Geometry/coupe")[`coupe`] library on NVIDIA GPUs.
]

#align(center, text(16pt)[CEA, DAM, DIF, F-91297, Arpajon, France])
#align(center, text(14pt)[August 31, 2023])
