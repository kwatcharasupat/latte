pip install .[tests] --quiet
python -m pytest -s tests/ --cov latte --cov-report term-missing

