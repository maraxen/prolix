SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

.PHONY: help clean html

help:
	@echo "Please use \`make <target>' where <target> is one of:"
	@echo "  html       build standalone HTML documentation into $(BUILDDIR)/html"
	@echo "  clean      remove Sphinx build artifacts"

clean:
	rm -rf $(BUILDDIR)/ build/

html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
