from scripts.constants import const
from scripts.loader import load_dataset
from scripts.preprocess import preprocess_dataset
from scripts.extraction import feature_extraction
from scripts.model import training_and_inference

# TODO: Remove warnings import before submissions..Below 2 lines
import warnings
warnings.filterwarnings("ignore")

# Path to data directory
data_dir = '../data/'

data_content = const(data_dir)
data_content = load_dataset(data_content)
data_content = preprocess_dataset(data_content)

# Prediction 1
data_content.time_zone_feat = False
data_content.best_params = {'objective': 'rank:map',
                            'learning_rate': 0.04,
                            'gamma': 1.0,
                            'booster': 'gbtree',
                            'max_depth': 5,
                            'n_estimators': 3250,
                            'reg_lambda': 0.7}

data_content = feature_extraction(data_content)
data_content = training_and_inference(data_content)
print('Prediction 1 saved successfully in current directory...\n', flush=True)

# Prediction 2
data_content.time_zone_feat = True
data_content.best_params = {'objective': 'rank:map',
                            'learning_rate': 0.04,
                            'gamma': 0.5,
                            'booster': 'gbtree',
                            'max_depth': 5,
                            'n_estimators': 2390,
                            'reg_lambda': 0.7}

data_content = feature_extraction(data_content)
data_content = training_and_inference(data_content)
print('Prediction 2 saved successfully in current directory...\n', flush=True)
