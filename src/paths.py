import os

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

VGGISH_DIR = os.path.join(ROOT_DIR, 'vggish')
DATA_DIR   = os.path.join(ROOT_DIR, 'data')

MODELS_DIR     = os.path.join(ROOT_DIR, 'models')
SVM_MODELS_DIR =  os.path.join(MODELS_DIR, 'svm')
MLP_MODELS_DIR =  os.path.join(MODELS_DIR, 'mlp')

VGGISH_PCA_PARAMS = os.path.join(VGGISH_DIR, 'vggish_pca_params.npz')
VGGISH_MODEL      = os.path.join(VGGISH_DIR, 'vggish_model.ckpt')
