#!/usr/bin/env bash

# Install dataset
wget https://zenodo.org/records/1432913/files/openmic-2018-v1.0.0.tgz\?download\=1

mkdir -p ./data/openmic
tar xvzf "openmic-2018-v1.0.0.tgz?download=1"

mv openmic-2018/* ./data/openmic
rm -r openmic-2018 openmic-2018-v1.0.0.tgz\?download=1

# Install vggish model weights
mkdir -p ./vggish

curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz

mv vggish_model.ckpt ./vggish
mv vggish_pca_params.npz ./vggish
