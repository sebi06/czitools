#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the "upload" functionality of this file, you must:
#   $ pipenv install twine --dev

import pathlib
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "czitools"
DESCRIPTION = "Tools to simplify reading CZI (Carl Zeiss Image) meta-and pixeldata in Python"
URL = "https://github.com/sebi06/czitools"
EMAIL = "sebrhode@gmail.com"
AUTHOR = "Sebastian Rhode"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "0.0.1"

# What packages are required for this module to be executed?
REQUIRED = [
    "setuptools>=52.0.0",
    "xmltodict>=0.12.0",
    "pydash>=5.0.2",
    "aicsimageio[all]",
    "aicspylibczi>=3.0.4",
    "tqdm>=4.61.2",
    "pylibczirw",
    "cztile",
]

# What packages are optional?
EXTRAS = {
    # "fancy feature": ["django"],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if "README.md" is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package"s __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system(
            "{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()

############ SETUP #################


setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author=AUTHOR,
      author_email=EMAIL,
      python_requires=REQUIRES_PYTHON,
      url=URL,
      # packages=["czitools",],
      packages=find_packages(),
      #packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
      # If your package is a single module, use this instead of "packages":
      # py_modules=["mypackage"],

      # entry_points={
      #     "console_scripts": ["mycli=mymodule:cli"],
      # },
      install_requires=REQUIRED,
      extras_require=EXTRAS,
      zip_safe=False,
      include_package_data=True,
      license="BSD 3-Clause License",
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering",
          "License :: OSI Approved :: BSD License",
          "Programming Language :: Python :: 3.9",
      ],

      # $ setup.py publish support.
      # cmdclass={
      #    "upload": UploadCommand,
      # },
      )
