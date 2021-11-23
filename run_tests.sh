export AUTOGRAPH_VERBOSITY=0
export TF_CPP_MIN_LOG_LEVEL=2
pip install .[tests] #--ignore-installed --no-binary :all: latte
python -m pytest -sv tests/ --cov latte --cov-report term-missing

