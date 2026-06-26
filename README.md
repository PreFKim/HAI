# HAI

# Environment

```
conda deactivate
conda env remove --name HAI -y
conda env create -f environment.yaml
conda activate HAI
pip install --upgrade pip

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install transformers==4.37.2 timm

git clone https://github.com/OpenGVLab/InternImage.git
cd InternImage/classification/ops_dcnv3
sh make.sh
python test.py
cd ../../../
cp -r ./InternImage/classification/ops_dcnv3 ./ops_dcnv3

git clone https://github.com/OpenGVLab/DCNv4.git
mv -f DCNv4 FlashInternImage2
cd ./FlashInternImage/DCNv4_op
bash make.sh
cd ../../
```
