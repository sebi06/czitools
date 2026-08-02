from czitools.export_tools import create_gui
import napari

viewer = napari.Viewer()
viewer.window.add_dock_widget(create_gui(), name="CZI OME-ZARR Converter")

napari.run()
