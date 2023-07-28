alias b := build
alias o := open

output_pdf := "master_thesis.pdf"

version := "v0.1.0"
tmpdir := "Gabriel_DOS_SANTOS-master_thesis"
tarball := tmpdir + "-" + version + ".tar.gz"

default:
	@just --list

build:
	typst compile src/main.typ --root . {{output_pdf}}

watch: open
	typst watch src/main.typ --root . {{output_pdf}}

open: build
	@open {{output_pdf}}

clean:
	@rm -f {{tarball}} {{output_pdf}}

compress:
	@echo "Generating a tarball..."
	@rm -f {{tarball}}
	@mkdir -p {{tmpdir}}
	@cp -r README.md LICENSE-* justfile src/* images/* {{tmpdir}}
	@tar czf {{tarball}} {{tmpdir}}
	@rm -rf {{tmpdir}}
