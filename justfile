alias b := build
alias c := clean
alias o := open

OUTPUT_PDF := "GABRIEL_DOS_SANTOS-MASTER_THESIS.pdf"

VERSION := "v0.1.0"
TMPDIR := "GABRIEL_DOS_SANTOS-MASTER_THESIS"
TARBALL := TMPDIR + "-" + VERSION + ".tar.gz"

default:
	@just --list

# Print the help
help: default

# Build the main PDF
build:
	typst compile src/main.typ --root . {{OUTPUT_PDF}}

# Watch and build the PDF on changes
watch: open
	typst watch src/main.typ --root . {{OUTPUT_PDF}}

# Open the PDF in a document viewer
open: build
	@okular {{OUTPUT_PDF}} &

# Clean the artifacts from the directory
clean:
	@rm -f {{TARBALL}} {{OUTPUT_PDF}}

# Create an archive of the directory
compress:
	@echo "Generating a TARBALL..."
	@rm -f {{TARBALL}}
	@mkdir -p {{TMPDIR}}
	@cp -r README.md LICENSE-* justfile src/* images/* {{TMPDIR}}
	@tar czf {{TARBALL}} {{TMPDIR}}
	@rm -rf {{TMPDIR}}
