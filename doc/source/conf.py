# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'pyNICAM-DC'
copyright = '2023, University of Tokyo'
author = 'Tomoki Miyakawa'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    # TODO: remove this once fixed upstream
    # see https://github.com/readthedocs/sphinx_rtd_theme/issues/1452
    'sphinxcontrib.jquery',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.automodule',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
