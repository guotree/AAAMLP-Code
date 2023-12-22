conda env remove --name ml
conda create -n ml python=3.7.6
conda activate ml
pip install -r requirements.txt