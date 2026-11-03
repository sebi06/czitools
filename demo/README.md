## Important Remarks

* This folder contains scripts to "play" around and try out new things.
* None of those script are part of the release or officially supported.
* They are not tested and might break at any time.
* They might require additional dependencies which are not part of the main package.
* Some require to install additional packages like bioio and bioio-czi

## CZI Vzip Examples

Install `czitools[vzip]` in the environment first. From the repository root:

```bash
python demo/scripts/create_czi_vzip.py image.czi
python demo/scripts/create_czi_vzip.py image.czi -o archives/image.vzip --overwrite
python demo/scripts/view_czi_vzip_napari.py image.vzip
```

The creation example uses the GUI-independent `create_czi_vzip` public API.
The viewing example additionally requires napari and Qt, and reopens the archive
as lazy image layers without zarrifying the CZI again. Use an activated
environment (for example `pixi run ...` in a configured Pixi workspace), rather
than launching an unactivated Conda interpreter directly on Windows.

Archives reference the original CZI, which must remain unchanged at its original
location. These examples do not copy pixels or implement a napari reader plugin.

## Disclaimer

* This content of this repository is free to use for everybody and purely experimental.
* Use them on your own risk.

**By using any of those examples you agree to this disclaimer.**

Version: 2026.08.02
