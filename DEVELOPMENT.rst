Development
===========

Build for Windows
-----------------

.. code-block:: python

    pyinstaller --onefile --windowed --add-data "xirescore/assets;xirescore/assets" --hidden-import pyarrow.vendored.version --name=xiRESCORE xirescore/__main__.py
