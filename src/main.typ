#set page(paper: "a4")
#set par(leading: 0.55em, first-line-indent: 1.8em, justify: true)

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

#set heading(numbering: "1.1.", outlined: true)

#include "chapters/1-introduction.typ"

#pagebreak()

#include "chapters/2-context.typ"

#pagebreak()

#include "chapters/3-state_of_the_art.typ"

#pagebreak()

#include "chapters/4-contribution.typ"

#pagebreak()

#include "chapters/5-conclusion.typ"

#pagebreak()

#include "chapters/6-bibliography.typ"

#pagebreak()

#include "chapters/7-appendix.typ"
