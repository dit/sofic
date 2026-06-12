# Minimal Makefile for Sphinx documentation

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR      = _build
ALLSPHINXOPTS = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) .

.PHONY: html linkcheck clean

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck

clean:
	-rm -rf $(BUILDDIR)/*
