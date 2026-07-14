"""Launch the CZI -> OME-Zarr converter GUI (czitools Stage 5).

Requires the GUI extra::

    pip install "czitools[omezarr-gui]"

Run from anywhere::

    python demo/scripts/run_omezarr_gui.py

Or use the installed console script::

    czitools-omezarr-gui

The GUI can also be embedded in napari::

    from czitools.export_tools import create_gui
    viewer.window.add_dock_widget(create_gui(), name="CZI Converter")
"""

from czitools.export_tools import run_gui

if __name__ == "__main__":
    run_gui()
