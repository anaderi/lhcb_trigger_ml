from __future__ import division, print_function, absolute_import
import io
__author__ = 'Alex Rogozhnikov'


def execute_notebook(filename):
    """Allows one to execute cell-by-cell some IPython notebook provided its name"""
    from IPython.core.getipython import get_ipython
    from IPython.nbformat import current

    with io.open(filename) as f:
        notebook = current.read(f, 'json')

    ip = get_ipython()
    for cell in notebook.worksheets[0].cells:
        if cell.cell_type == 'code':
            ip.run_cell(cell.input)


def test_notebooks():
    # Setting flag for
    # debug_notebook = True
    # execute_notebook('../notebooks/DemoMetrics.ipynb')
    # https://github.com/ipython/ipython/wiki/Cookbook:-Notebook-utilities
    # https://gist.github.com/minrk/2620735
    pass