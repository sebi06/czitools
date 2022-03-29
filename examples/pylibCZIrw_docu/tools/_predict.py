# -*- coding: utf-8 -*-

#################################################################
# File        : _predict.py
# Version     : 0.0.2
# Author      : Team Enchilada
# Date        : 13.01.2022
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import numpy as np
import tempfile
import itertools
from typing import List, Tuple, Dict, Union
from pathlib import Path
import dask.array as da
import json
from napari_czmodel_segment import _extract_model, _tiling
from ._onnx_inference import OnnxInferencer


def predict(input_batch: List[np.ndarray], model_path) -> List[np.ndarray]:

    inferencer = OnnxInferencer(str(model_path))
    prediction_batch = inferencer.predict(input_batch)

    return prediction_batch


def get_tilesize_prediction(model_path: str) -> Tuple[int, int, int, int]:

    inferencer = OnnxInferencer(str(model_path))
    tilesize_prediction = inferencer.get_input_shape()

    return tilesize_prediction[1], tilesize_prediction[2]


def predict_tile(tile2d: Union[np.ndarray, da.Array],
                 model_path: str) -> Union[np.ndarray, da.Array]:

    # make sure an numpy array is used for the prediction
    if isinstance(tile2d, da.Array):
        tile2d = tile2d.compute()

    # get the prediction for that tile
    pred_raw_tile = predict([tile2d[..., np.newaxis]], model_path)

    # get the labels and add 1 to reflect the real values
    pred_tile = np.argmax(pred_raw_tile[0], axis=-1) + 1

    return pred_tile


def predict_ndarray(czann_file: str,
                    img: Union[np.ndarray, da.Array],
                    output_dask: bool = True,
                    border: Union[str, int] = "auto",
                    verbose: bool = True) -> Tuple[List, List, int, Union[np.ndarray, da.Array]]:

    if output_dask:
        seg_complete = da.zeros_like(img, chunks=img.shape)
    if not output_dask:
        seg_complete = np.zeros_like(img)

    # get the shape without the XY dimensions
    shape_woXY = img.shape[:-2]

    # create the "values" each for-loop iterates over
    loopover = [range(s) for s in shape_woXY]
    prod = itertools.product(*loopover)

    # extract the model information and path and to the prediction
    with tempfile.TemporaryDirectory() as temp_path:

        # get the model metadata and the path
        model_metadata, model_path = _extract_model.extract_model(czann_file, Path(temp_path))

        req_tilewidth, req_tileheight = get_tilesize_prediction(model_path)
        print("Used TileSize: ", req_tilewidth, req_tileheight)
        if verbose:
            # print the metadata "pretty"
            pretty_model_metadata = json.dumps(model_metadata, indent=4, sort_keys=False)
            print(pretty_model_metadata)

        # create variables
        classnames = {}
        labelvalues = {}
        numclasses = len(model_metadata["Model"]["TrainingClasses"]["Item"])

        # extract some info from the model metadata to make it "easier" to read the code
        for c in range(numclasses):
            classnames[c] = model_metadata["Model"]["TrainingClasses"]["Item"][c]["@Name"]
            labelvalues[c] = int(model_metadata["Model"]["TrainingClasses"]["Item"][c]["@LabelValue"])

        # get the used bordersize - is needed for the tiling
        if isinstance(border, str) and border == "auto":
            bordersize = int(model_metadata["Model"]["BorderSize"])
        else:
            bordersize = border

        print("Used Minimum BorderSize for Tiling: ", bordersize)

        # loop over all dimensions
        for idx in prod:

            # create list of slice-like objects based on the shape_woXY
            sl = len(shape_woXY) * [np.s_[0:1]]

            # insert the correct index into the respective slice objects for all dimensions
            for nd in range(len(shape_woXY)):
                sl[nd] = idx[nd]

            # extract the 2D image from the n-dims stack using the list of slice objects
            img2d = np.squeeze(img[tuple(sl)])

            # process the whole 2d image - make sure to use the correct **kwargs
            new_img2d = _tiling.process2d_tiles(predict_tile,
                                                img2d,
                                                tile_width=req_tilewidth,
                                                tile_height=req_tileheight,
                                                min_border_width=bordersize,
                                                use_dask=False,
                                                model_path=model_path)

            # insert new 2D after tile-wise processing into nd array
            seg_complete[tuple(sl)] = new_img2d

    return classnames, labelvalues, numclasses, seg_complete
