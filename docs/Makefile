.PHONY: help clean html

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  clean      to remove build artifacts"

clean:
	rm -rf build/

html:
	sphinx-build -b html source build/html
	@echo
	@echo "Build finished. The HTML pages are in build/html."
