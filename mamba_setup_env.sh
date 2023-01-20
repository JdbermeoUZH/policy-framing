mamba create -n Framing_py39 python=3.9.13
source activate Framing_py39

mamba install pandas -y
mamba install matplotlib -y
mamba install scikit-learn -y
mamba install -c conda-forge spacy -y
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download pl_core_news_sm
python -m spacy download ru_core_news_sm
mamba install -c conda-forge py-xgboost -y
mamba install -c conda-forge scikit-optimize -y
mamba install -c conda-forge imbalanced-learn -y
pip install PyYAML
pip install mlflow-skinny
pip install iterative-stratification
pip install termcolor
