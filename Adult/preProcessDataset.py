from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

url = sys.argv[1]
split = float(sys.argv[2])
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#s = requests.get(url).content
names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
         "relationship", "race", "sex",
         "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
# Importing the datasets
dataset = pd.read_csv(url, names=names)     #dataset = pd.read_csv(io.StringIO(s.decode('utf-8')), names=names)

# mark ? values as missing or NaN
dataset.replace(to_replace="[?]", value=np.nan, regex=True, inplace=True)

dataset = dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 14].values

df = pd.DataFrame(dataset)

label_encoder = LabelEncoder()
df["workclass"] = label_encoder.fit_transform(df["workclass"])
df["education"] = label_encoder.fit_transform(df["education"])
df["marital-status"] = label_encoder.fit_transform(df["marital-status"])
df["occupation"] = label_encoder.fit_transform(df["occupation"])
df["relationship"] = label_encoder.fit_transform(df["relationship"])
df["race"] = label_encoder.fit_transform(df["race"])
df["sex"] = label_encoder.fit_transform(df["sex"])
df["native-country"] = label_encoder.fit_transform(df["native-country"])
df["class"] = label_encoder.fit_transform(df["class"])

# standardizing the dataset
df["age"] = preprocessing.scale(df["age"])
df["workclass"] = preprocessing.scale(df["workclass"])
df["fnlwgt"] = preprocessing.scale(df["fnlwgt"])
df["education"] = preprocessing.scale(df["education"])
df["education-num"] = preprocessing.scale(df["education-num"])
df["marital-status"] = preprocessing.scale(df["marital-status"])
df["occupation"] = preprocessing.scale(df["occupation"])
df["relationship"] = preprocessing.scale(df["relationship"])
df["race"] = preprocessing.scale(df["race"])
df["sex"] = preprocessing.scale(df["sex"])
df["capital-gain"] = preprocessing.scale(df["capital-gain"])
df["capital-loss"] = preprocessing.scale(df["capital-loss"])
df["hours-per-week"] = preprocessing.scale(df["hours-per-week"])
df["native-country"] = preprocessing.scale(df["native-country"])

# writing pre-processed data to a csv file
df.to_csv('preProcessedAdultDataset.csv', sep=',', index=False, header=False)

# split dataset in to train and test
train,test = train_test_split(df, test_size=split)
train.to_csv('train.csv', sep=',', index=False, header=False)
test.to_csv('test.csv', sep=',', index=False, header=False)