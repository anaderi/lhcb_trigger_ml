# This is supplementary function, which connects other IPython notebook
# with helpful functions. 
# Absolutely inimportant
import io
from IPython.nbformat import current

def execute_notebook(fileName):
    with io.open(fileName) as f:
        nb = current.read(f, 'json')
    
    ip = get_ipython()
    for cell in nb.worksheets[0].cells:
        if cell.cell_type != 'code': continue
        ip.run_cell(cell.input)
