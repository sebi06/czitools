import numpy as np
from czitools.metadata_tools.czi_metadata import convert_numpy_types


def test_convert_numpy_scalar():
    scalar = np.int32(42)
    result = convert_numpy_types(scalar)
    assert isinstance(result, int)
    assert result == 42


def test_convert_numpy_array():
    array = np.array([1, 2, 3])
    result = convert_numpy_types(array)
    assert isinstance(result, list)
    assert result == [1, 2, 3]


def test_convert_nested_numpy_array():
    nested_array = np.array([[1, 2], [3, 4]])
    result = convert_numpy_types(nested_array)
    assert isinstance(result, list)
    assert result == [[1, 2], [3, 4]]


def test_convert_dict_with_numpy():
    data = {"a": np.int32(10), "b": np.array([1, 2, 3])}
    result = convert_numpy_types(data)
    assert isinstance(result, dict)
    assert result["a"] == 10
    assert result["b"] == [1, 2, 3]


def test_convert_list_with_numpy():
    data = [np.int32(5), np.array([1, 2, 3])]
    result = convert_numpy_types(data)
    assert isinstance(result, list)
    assert result[0] == 5
    assert result[1] == [1, 2, 3]


def test_convert_tuple_with_numpy():
    data = (np.float64(3.14), np.array([4, 5, 6]))
    result = convert_numpy_types(data)
    assert isinstance(result, tuple)
    assert result[0] == 3.14
    assert result[1] == [4, 5, 6]


def test_convert_mixed_structure():
    data = {
        "key1": np.array([1, 2, 3]),
        "key2": [np.int32(10), {"nested": np.float64(2.71)}],
    }
    result = convert_numpy_types(data)
    assert isinstance(result, dict)
    assert result["key1"] == [1, 2, 3]
    assert result["key2"][0] == 10
    assert result["key2"][1]["nested"] == 2.71


def test_convert_non_numpy_object():
    data = "string"
    result = convert_numpy_types(data)
    assert result == "string"
