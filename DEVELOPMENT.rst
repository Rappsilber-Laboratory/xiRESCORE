Development
===========

Build for Windows
-----------------

.. code-block:: bash

    $ pyinstaller --onefile --windowed --add-data "xirescore/assets;xirescore/assets" --hidden-import pyarrow.vendored.version --icon=xirescore/assets/xirescore_logo.ico --name=xiRESCORE xirescore/__main__.py

.. hint::

    Windows 8 has binaries incompatible with pyarrow 16 and above. Use Python 3.10 to be able to pip install
    pre-compiled pyarrow versions.
