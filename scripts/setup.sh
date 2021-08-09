apt install -y vim screen git-lfs

# Download kaggle competition dataset.
pip install --upgrade kaggle

export KAGGLE_USERNAME=[your kaggle username]
export KAGGLE_KEY=[your kaggle key]

kaggle competitions download -c commonlitreadabilityprize
unzip -qq commonlitreadabilityprize.zip -d commonlit-readability-prize
rm commonlitreadabilityprize.zip

# Install the requirements and login to wandb.
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
wandb login [your wandb key]

# Install NVIDIA apex library.
git clone https://github.com/NVIDIA/apex
sed -i "s/or (bare_metal_minor != torch_binary_minor)//g" apex/setup.py
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/
rm -rf apex