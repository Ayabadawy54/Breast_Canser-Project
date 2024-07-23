
## Major Libraries
import pandas as pd
import os
## sklearn -- for pipeline and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
#from sklearn_features.transformers import DataFrameSelector # it case a lot of problems ahhhhh :(

df = pd.read_csv("Breast_canser_data.csv")
df

df = df.drop(['Unnamed: 32','id'],axis=1)
df

df=df.rename(columns={"diagnosis":"target"})
 #B:non-cancerous   , M:cancerous

label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

X = df.drop(columns=['target'], axis=1)   ## Features
y = df['target']   ## target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

num_cols = [col for col in X_train.columns if X_train[col].dtype in ['float32', 'float64', 'int32', 'int64']]
categ_cols = [col for col in X_train.columns if X_train[col].dtype not in ['float32', 'float64', 'int32', 'int64']]
num_cols

# Building a pipeline for numerical variables
num_pipeline = Pipeline(steps=[
                        ('scaler', StandardScaler())
                              ]
                       )
## deal with (num_pipline) as an instance -- fit and transform to train dataset and transform only to other datasets
X_train_final = num_pipeline.fit_transform(X_train[num_cols])  ## train

def preprocess_new(X_new):
    ''' This Function tries to process the new instances before predicted using Model
    Args:
    *****
        (X_new: 2D array) --> The Features in the same order
            ['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concave points_mean',
 'symmetry_mean',
 'fractal_dimension_mean',
 'radius_se',
 'texture_se',
 'perimeter_se',
 'area_se',
 'smoothness_se',
 'compactness_se',
 'concavity_se',
 'concave points_se',
 'symmetry_se',
 'fractal_dimension_se',
 'radius_worst',
 'texture_worst',
 'perimeter_worst',
 'area_worst',
 'smoothness_worst',
 'compactness_worst',
 'concavity_worst',
 'concave points_worst',
 'symmetry_worst',
 'fractal_dimension_worst']
        All Featutes are Numerical

     Returns:
     *******
         Preprocessed Features ready to make inference by the Model
    '''
    return num_pipeline.transform(X_new)
