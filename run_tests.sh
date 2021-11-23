pip install .[tests] #--ignore-installed --no-binary :all: latte
python -m pytest -sv tests/ --cov latte --cov-report term-missing

