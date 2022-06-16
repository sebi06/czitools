python -m pip install --upgrade pip

python -m pip install --user --upgrade setuptools wheel

python -m pip install --upgrade build

python -m build

python setup.py sdist bdist_wheel

python -m pip install --user --upgrade twine

#REM python -m twine upload --repository pypi dist/*
python -m twine upload --repository testpypi dist/*
