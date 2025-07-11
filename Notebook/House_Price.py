# %% [markdown]
# **Table of contents**<a id='toc0_'></a>    
# - [model](#toc1_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %%
import pandas as pd
import numpy as np
import re
from IPython.display import display, HTML
import sys
import os

import warnings
warnings.filterwarnings("ignore")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

display(HTML("<style>.container { width:50% !important; }</style>"))
#display(HTML('<style>.output { max-width:800px !important; }</style>')) # control output width
display(HTML("<style>.container { width:130% !important; }</style>"))

pd.set_option('display.max_columns', None)   # Show number of columns, None means show all columns
pd.set_option('display.max_rows', 50)        # show top and bottom 15, middle with ......
pd.set_option('display.width', None)         # Set the width of the display
pd.set_option('display.max_colwidth', None)  # Show full content of each column

from sklearn import datasets as sklearn_datasets
from sklearn import preprocessing
from sklearn import svm
from sklearn import neighbors
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.feature_selection import mutual_info_classif as MIC # the most relevant correlation to target variable, useful for classification problem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold,GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score  # Ensure recall_score is imported
from sklearn.metrics import log_loss

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score

from sklearn.metrics import ConfusionMatrixDisplay



from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler      # base on each column/feature, it has mean 0 and SD 1
from sklearn.preprocessing import MinMaxScaler        # minimum value is 0 and maximum value is 1
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder 

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MaxAbsScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.utils.class_weight import compute_sample_weight

from catboost import CatBoostClassifier # good for classifcation
from catboost import CatBoostRegressor

import xgboost as xgb
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor


# %%


# %%
# house_price = pd.read_csv('https://raw.githubusercontent.com/KevinJianLin/House_Price_Advanced_Regression/refs/heads/main/Data/train.csv')
filepath1 = '/Users/jianbinlin/Documents/Scientific_2022_Jan.csv'
house_ON = pd.read_csv(filepath1).iloc[5:,104:131].reset_index(drop=True)
house_ON.columns = house_ON.iloc[0]
house_ON = house_ON[1:].reset_index(drop=True)
house_unionville = house_ON.copy()
house_ON = house_ON[(house_ON['comments'].str.contains("relevant")) &  ~(house_ON['comments'].isna()) & ~(house_ON['sold price\n(million)'].isna()) ].reset_index(drop=True)
new_data = house_unionville[(house_unionville['comments'].str.contains("relevant")) &  ~(house_unionville['comments'].isna()) & (house_unionville['sold price\n(million)'].isna()) ].reset_index(drop=True)

# house_ON["sold price\n(million)"] = house_ON[['asking price\n(million)', 'sold price\n(million)']].apply(
#     lambda x: x['sold price\n(million)'] if not pd.isna(x['sold price\n(million)']) else x['asking price\n(million)'], axis=1)
int_cols = ['built age', 'Kitchen', 'bathroom', 'Bedrooms', 'Rooms','Parking Drive Spaces','Garage Parking Space']
float_cols = ['sold price\n(million)', 'lot size \n(feet 2)']
bool_cols = []
drop_cols = ['Listing Days', 'asking price\n(million)','sold price\n(million)','sqft/price','comments','Municipality']
house_ON[int_cols] = house_ON[int_cols].fillna(0).astype(int)
house_ON[float_cols] = house_ON[float_cols].astype(float) 
house_ON['sold year'] = pd.to_datetime(house_ON['sold year'])
# house_ON['sold year'] = house_ON['sold year'].dt.to_period('M').dt.to_timestamp()
# house_ON['sold year'] = house_ON['sold year'].dt.strftime('%Y-%m')
house_ON = house_ON.bfill().ffill()  # fill missing values with backfill and forward fill
y = house_ON['sold price\n(million)']

house_ON.drop(columns=drop_cols, inplace=True)  # drop comments column
# house_ON.dropna(subset=['sold price\n(million)'], inplace=True)  # drop rows with no sold price



data_set = house_ON.copy()
data_set.dtypes
data_set.shape
data_set.head(5)    

# %%
new_data[int_cols] = new_data[int_cols].fillna(0).astype(int)
new_data[float_cols] = new_data[float_cols].astype(float) 
new_data_visual = new_data.copy()
new_data = new_data.bfill().ffill()  # fill missing values with backfill and forward fill
drop_cols_new= drop_cols + ['sold year']
new_data.drop(columns=drop_cols_new, inplace=True)  # drop comments column
new_data.dtypes
new_data

# %%
class data_profiling:
    def __init__(self,*args):
        self.df       = args[0]
        if len(args) >1 and args[1]:
            self.cat_col  = args[1]
        else:
            self.cat_col = [col for col in self.df.columns if len(self.df[col].unique()) < 40 ]
        self.float_column = [col for col in self.df.columns if self.df[col].dtype == float]
        self.int_column = [col for col in self.df.columns if self.df[col].dtype == int]
        self.date_columns  = []
        self.rest_columns  = [col for col in self.df.columns if col not in (self.float_column + self.int_column + self.cat_col)]
        self.col_min_char = pd.DataFrame(self.df.astype(str).apply(lambda x:x.str.len().min()),columns=['min_char'])
        self.col_max_char = pd.DataFrame(self.df.astype(str).apply(lambda x:x.str.len().max()),columns=['max_char'])
        self.term_deposit_non_ascii     = pd.DataFrame(self.df.apply(lambda x: sum(ord(char)>127 for chars in x for char in str(chars)),axis=0),columns=['non-ascii character'])
        self.term_deposit_null_value    = pd.DataFrame(self.df.isna().sum(),columns=['number of nan and none values'])
        self.size_mega        = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        self.number_of_duplicated_rows  = sum(self.df.duplicated())
        self.space = []
        self.variance = 'pending compute'
        self.skew     = 'pending compute'
        self.kurtosis = 'pending compute'
        self.z_score  = 'pending compute'
        

        for col in self.df.columns:
            if self.df[col].dtype == 'object':  # Check if the column is of string type
                total_spaces = self.df[col].apply(lambda x: self.count_empty_space(x) if x is not (np.nan or None) else 0).sum() 
            else:total_spaces=0
            self.space.append(total_spaces)    
        self.empty_string =[]
        
    def count_empty_space(self,x):
        return len(re.findall(r' +', x))
    def __call__(self):
        le = LabelEncoder()
        term_deposit_encoded = self.df.copy()
        for col in self.cat_col:
            term_deposit_encoded[col] = le.fit_transform(self.df[col])
        self.term_deposit_describe = pd.DataFrame(term_deposit_encoded.describe().transpose())
        self.duplicated_rows = pd.DataFrame({"duplicated_rows":[sum(self.df.duplicated())]*self.df.shape[1]},index=self.df.columns.to_list())
        self.duplicated_cols = pd.DataFrame({"duplicated_columnss":[sum(self.df.transpose().duplicated())]*self.df.shape[1]},index=self.df.columns.to_list())

        self.shape_size = pd.DataFrame({"shape and size":[[self.df.shape]+["{:.2f} Mb".format(self.size_mega)]]*self.df.shape[1]},index=self.df.columns.to_list())
        self.empty_string_total = pd.DataFrame({"Completeness_Empty":[self.empty_string]},index=self.df.columns.to_list())
        self.empty_space_total =  pd.DataFrame({"Completeness_Space":[ self.space]*self.df.shape[1]},index=self.df.columns.to_list())

        self.float_col =  pd.DataFrame({"float_col":[ self.float_column]*self.df.shape[1]},index=self.df.columns.to_list())
        self.float_col_length =  pd.DataFrame({"float_col_legth":[ len(self.float_column)]*self.df.shape[1]},index=self.df.columns.to_list())

        self.int_col =  pd.DataFrame({"int_col":[ self.int_column]*self.df.shape[1]},index=self.df.columns.to_list())
        self.int_col_length =  pd.DataFrame({"int_col_length":[ len(self.int_column)]*self.df.shape[1]},index=self.df.columns.to_list())
    
        return pd.concat([self.term_deposit_describe,self.term_deposit_null_value,self.col_min_char,self.col_max_char,self.term_deposit_non_ascii,
                          self.duplicated_rows,self.duplicated_cols,self.shape_size,self.empty_string_total, self.empty_space_total,self.float_col,
                          self.float_col_length,self.int_col,self.int_col_length],axis=1)
    
    def __repr__(self):
        """
        stands for representation
        print(repr(data_profiling))
        """
        return self.__class__.__name__


# %%
data_profile = data_profiling(data_set)
# data_profile()
# data_profile.cat_col
# data_profile.float_column
# data_set[data_profile.int_column]
# target_variable = 'SalePrice'
# target_variable = 'sold price\n(million)'
# X = data_set[data_profile.int_column].drop(columns =[target_variable])
X = house_ON.copy()
# y = house_price[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)



# %%


# %%


# %%


# %% [markdown]
# ### 

# %% [markdown]
# # <a id='toc1_'></a>[model](#toc0_)

# %%
# data_profile.int_column.remove(target_variable)

preprocessor = ColumnTransformer(transformers = [
                                   #         ('text', TfidfVectorizer(max_features=500), text_col),
                                               ('int_col',StandardScaler(),data_profile.int_column),
                                          #('int_col',StandardScaler(), data_profile.int_column[3:-1]),
                                          # ('int_col',StandardScaler(), data_profile.int_column[0:3]),
                                        #    ('cat_col', OneHotEncoder(), cat_col),
                                            ]) 

# %%
model_parameters_regressor = {
    'decisiontree_regressor':{
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'regressor__max_depth': [None, 1, 2], # one feature can be used more than once...[None, 10, 20, 30]
            'regressor__min_samples_split': [2, 3], # mimimal data points of node spliting default is 2.....[2, 5, 10]
            'regressor__min_samples_leaf': [1, 2] # leaf node is end point of a branch; default value is 1...[1, 2, 4]
        }
    },
    'randomforest_regressor':{
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [10, 20, 30], # number of decision tree,[100, 200, 300], larger number, more accurate but higher computational time
            'regressor__criterion': ['squared_error', 'absolute_error'], # default is squared_error..['squared_error', 'absolute_error', 'poisson']
            'regressor__max_depth': [None, 20], # maximum number of tree, None mean no limit
            'regressor__bootstrap': [True, False], # whether to use bootstrapping for sampling data when building tree
            'regressor__max_features': ['sqrt', None], # None means use all feature,['sqrt', 'log2', None]
            'regressor__max_samples': [0.1, 0.1] #fraction of sample to fit the tree,[0.1, 1.0]
        }
    },
    'adaboost_regressor':{
        'model': AdaBoostRegressor(),
        'params': { 
            'regressor__n_estimators': [50, 100], # default is 50, number of weak learner [50, 100, 200]
            'regressor__learning_rate': [0.5, 1], # lower value, hard to converge,[0.01, 0.1, 0.5, 1]  
            'regressor__loss': ['linear', 'square'] # default is linear 'regressor__loss': ['linear', 'square', 'exponential']
        }
    },
    'catboost_regressor':{
        'model': CatBoostRegressor(silent=True),
        'params': {
            'regressor__depth': [4, 6],# [4, 6, 8]
            'regressor__learning_rate': [0.01, 0.03], # [0.01, 0.03, 0.1]
            'regressor__l2_leaf_reg': [1, 3, 5] # default is 3, L2 regularization term on leaf weights, [1, 3, 5, 10]
                                                    #Helps control overfitting by penalizing large weights,Higher values make the model more conservative.
        }
    },
    'xgboost_regressor':{
        'model': xgb.XGBRegressor(),
        'params': {
            'regressor__max_depth': [3, 5],# [3, 5, 7]
            'regressor__min_child_weight': [1, 3], # Minimum sum of weights (or instances) required to create a leaf node. default is 1
                                                      # [1, 3, 5]      
            'regressor__gamma': [0, 0.1], # Minimum loss reduction required to make a split.
                                               # default is 0 ,[0, 0.1, 0.2]
            'regressor__subsample': [0.8, 1.0], #Fraction of samples used for training each tree.

            'regressor__colsample_bytree': [0.8, 1],# Fraction of features (columns) used for building each tree.[0.8, 1]

            'regressor__eta': [0.01, 0.5] #Step size shrinkage used to prevent overfitting.learning rate. default is 0.3. [0.01, 0.1]
        }
    },
    'lgb_regressor':{
        'model': LGBMRegressor(verbose=-1),
        'params': {
            'regressor__num_leaves': [31, 43], # Maximum number of leaves in each decision tree.Default: 31. [15, 31, 63]
            'regressor__max_depth': [3, 5], # -1 means tree to grow without limit.[3, 5, 7, -1]
            'regressor__learning_rate': [0.05, 0.1], #[0.01, 0.05, 0.1] 
            'regressor__min_child_samples': [10, 20], # Minimum number of data samples required in a leaf node. 
            'regressor__feature_fraction': [0.5, 0.7], # Proportion of features (columns) randomly selected for training each tree.
            'regressor__colsample_bytree': [None], #If None, feature_fraction takes precedence.
            'regressor__bagging_fraction': [0.5, 0.7] #Fraction of data randomly selected to train each tree (row sampling).default 1
        }
    },
    'mlp_regressor':{
        'model': MLPRegressor(max_iter=500, tol=1e-1, random_state=42, early_stopping=True),
        'params': {
            'regressor__hidden_layer_sizes': [(50,),    (50, 30)], # [(50,), (100,), (50, 30), (100, 50, 30)]
                                                                                       # (50,): A single hidden layer with 50 neurons.
                                                                                       # (100, 50, 30): Three hidden layers with 100, 50, and 30 neurons respectively.

            'regressor__activation': ['relu', 'tanh'], # Activation function for the hidden layers.
            'regressor__solver': ['adam', 'sgd'],  # Optimization algorithm used for training
            'regressor__alpha': [1e-5, 1e-4], # L2 penalty (regularization term)[1e-5, 1e-4, 1e-3, 1e-2]
            'regressor__learning_rate_init': [0.001, 0.01], # Initial learning rate, [0.0001, 0.001]
            'regressor__max_iter': [200, 300, 500], # 
            'regressor__early_stopping': [True],
            'regressor__n_iter_no_change': [10]  # Add early stopping patience
        }
    }
}


# %%


# %%

best_model ={}
results    = {}

for model_name, model_infor in model_parameters_regressor.items():
    pipeline = Pipeline([('preprocessor',preprocessor),
                         ('regressor',model_infor['model'])])
    grid_search = GridSearchCV(estimator=pipeline,
                                param_grid = model_infor['params'],
                                cv = 5, # 5 fold cv
                                scoring = 'neg_root_mean_squared_error', # neg_root_mean_squared_error,neg_mean_squared_error,r2
                                verbose =0, # not to display output
                                n_jobs = -1, # use all of cores
                              #   refit=False, # do not refit the model by the best parameters
                                )
    

    try:
      model_trained = grid_search.fit(X_train, y_train)
      best_model[model_name] = model_trained.best_estimator_
      #print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
      y_pred = model_trained.predict(X_test)
      rmse = mean_squared_error(y_test, y_pred)  # Test RMSE
      r2 = r2_score(y_test, y_pred)  # R^2 Score
      mae = mean_absolute_error(y_test, y_pred) # Mean Absolute Error
      mape = (abs((y_test - y_pred) / y_test).mean()) * 100 # Mean Absolute Percentage Error (MAPE)
      msle = mean_squared_log_error(y_test, y_pred)


      results[model_name] = {'MAE': mae,
                              'MAPE': mape,
                              'MSLE':msle,
                              'RMSE': rmse,
                              'R2': r2,
                              'Best Params': grid_search.best_params_
                              }
    except Exception as e:
      print(f"⚠️ Model {model_name} failed: {e}")
      results[model_name] = {
            'MAE': None,
            'MAPE': None,
            'RMSE': None,
            'Best Params': None,
            'Error': str(e)
        }   

    results_df = pd.DataFrame(results).T  # Transpose for readability

results_df

# %%
catboost_model = best_model['catboost_regressor']  # or whatever your CatBoost key is

y_new_pred = catboost_model.predict(new_data)   


# %%
y_new_pred

# %%
new_data_visual

# %%
new_data_visual.insert(new_data_visual.columns.get_loc('asking price\n(million)') + 1, 'predicted sold price\n(million)', y_new_pred)

# %%
new_data_visual

# %%


# %%


# %%



