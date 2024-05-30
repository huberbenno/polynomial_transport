import os
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

def test_notebooks():
    """
    Test a list of Jupyter notebooks by executing them and reporting any errors.

    The notebooks are specified in a list within this function.
    """
    notebook_list = [
        'x_rosenbrock.ipynb'
    ]

    for notebook_filename in notebook_list:
        notebook_path = os.path.join('../notebooks', notebook_filename)

        try:
            with open(notebook_path) as f:
                notebook_content = nbformat.read(f, as_version=4)

            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

            ep.preprocess(notebook_content, {'metadata': {'path': '../notebooks/'}})

        except CellExecutionError as e:
            pytest.fail(f"Notebook {notebook_filename} failed during execution: {str(e)}")
        except Exception as e:
            pytest.fail(f"Notebook {notebook_filename} failed with error: {str(e)}")
