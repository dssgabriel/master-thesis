// Entry file
#set page(paper: "a4")
#set par(leading: 0.55em, first-line-indent: 1.8em, justify: true)
#set list(indent: 0.8em)
#set enum(indent: 0.8em)
#set figure(numbering: "1-1")

#include "titlepage.typ"

#pagebreak()
#set page(numbering: "1 / 1")
#pagebreak()

#outline(indent: true, fill: repeat[` ·`], title: "Table of contents")

#pagebreak()
// Blank page
#pagebreak()

#set heading(numbering: none, outlined: false)
#include "acknowledgments.typ"

#pagebreak()

#include "cea.typ"

#pagebreak()

#set heading(numbering: "1.1", outlined: true)
#include "chapters/1-introduction.typ"

#pagebreak()

#include "chapters/2-context.typ"

#pagebreak()

#include "chapters/3-contributions.typ"

#pagebreak()

#include "chapters/4-conclusion.typ"

#pagebreak()

#include "chapters/5-bibliography.typ"

#pagebreak()

#include "chapters/6-appendix.typ"
