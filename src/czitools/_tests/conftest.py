import pytest
import pandas as pd

# Predefined planetable dict fixture for tests
PLANETABLE_DICT = {
    'Subblock': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    'S': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    'M': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    'T': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    'C': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    'Z': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    'X[micron]': {0: 16977.153, 1: 16977.153, 2: 16977.153, 3: 16977.153, 4: 16977.153},
    'Y[micron]': {0: 18621.489, 1: 18621.489, 2: 18621.489, 3: 18621.489, 4: 18621.489},
    'Z[micron]': {0: 1114.49, 1: 1114.81, 2: 1115.13, 3: 1115.45, 4: 1115.77},
    'Time[s]': {0: 0.0, 1: 0.894, 2: 1.792, 3: 2.681, 4: 3.586},
    'xstart': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    'ystart': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    'width': {0: 240, 1: 240, 2: 240, 3: 240, 4: 240},
    'height': {0: 170, 1: 170, 2: 170, 3: 170, 4: 170},
}


@pytest.fixture
def planetable_dict():
    """Return the predefined planetable dictionary."""
    return PLANETABLE_DICT


@pytest.fixture(autouse=True)
def _attach_planetable_to_unittest_instance(request, planetable_dict):
    """If a unittest.TestCase is running, attach the dict as self.planetable_dict so
    legacy unittest tests can access it as an attribute.
    This fixture is autouse so it runs for tests in this directory.
    """
    instance = getattr(request.node, "instance", None)
    if instance is not None:
        setattr(instance, "planetable_dict", planetable_dict)
    return planetable_dict


@pytest.fixture
def df():
    """Simple DataFrame used by tests that expect Time [s] and Value columns."""
    return pd.DataFrame({"Time [s]": [0, 1, 2, 3], "Value": [10, 20, 30, 40]})


@pytest.fixture
def planetable(planetable_dict):
    """Return a pandas DataFrame built from the predefined planetable dict."""
    return pd.DataFrame(planetable_dict)
