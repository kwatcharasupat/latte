pip install .[tests] --quiet
python -m pytest -sv tests/ --cov latte --cov-report term-missing

