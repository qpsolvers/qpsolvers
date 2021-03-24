# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR      = build
ONLINEDIR     = scaron.info:public_html/doc/qpsolvers
PDFFILE       = qpsolvers.pdf

# Internal variables.
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) src

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  linkcheck  to check all external links for integrity"

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*

.PHONY: html
html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

.PHONY: latexpdf
latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

.PHONY: linkcheck
linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

.PHONY: pdfopen
pdfopen: latexpdf
	zathura $(BUILDDIR)/latex/$(PDFFILE)

.PHONY: open
open:
	firefox $(BUILDDIR)/html/index.html

.PHONY: upload
upload: html latexpdf
	cp $(BUILDDIR)/latex/$(PDFFILE) $(BUILDDIR)/html/$(PDFFILE)
	rsync -auvz --delete-after $(BUILDDIR)/html/ $(ONLINEDIR)/
