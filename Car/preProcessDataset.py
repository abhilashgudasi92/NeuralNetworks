from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

url = sys.argv[1]
split = float(sys.argv[2])
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
# Importing the datasets
dataset = pd.read_csv(url, names=names)

# mark ? values as missing or NaN
dataset.replace(to_replace="[?]", value=np.nan, regex=True, inplace=True)
dataset = dataset.dropna()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

df = pd.DataFrame(dataset)
buying_type = ['low', 'med', 'high', 'vhigh']
maint_type = ['low', 'med', 'high', 'vhigh']
doors_type = [2, 3, 4, '5-more']
persons_type = [2, 4, 'more']
lug_boot_type = ['small', 'med', 'big']
safety_type = ['low', 'med', 'high']
class_type = ['unacc', 'acc', 'good', 'vgood']

df["buying"] = df["buying"].astype("category",
                                   categories=buying_type).cat.codes  # replacing buying_type values to [0,1,2,3]
df["maint"] = df["maint"].astype("category",
                                 categories=maint_type).cat.codes  # replacing maint_type values to [0,1,2,3]
df["lug_boot"] = df["lug_boot"].astype("category",
                                       categories=lug_boot_type).cat.codes  # replacing lug_boot_type values to [0,1,2]
df["safety"] = df["safety"].astype("category",
                                   categories=safety_type).cat.codes  # replacing safety_type values to [0,1,2]
df["class"] = df["class"].astype("category",
                                 categories=class_type).cat.codes  # replacing class_type values to [0,1,2,3]

df['doors'].replace('5more', 5, inplace=True)  # replacing '5more' as 5
df['persons'].replace('more', 5, inplace=True)  # replacing 'more' as 6

# standardizing the dataset
df["buying"] = preprocessing.scale(df["buying"])
df["maint"] = preprocessing.scale(df["maint"])
df["doors"] = preprocessing.scale(df["doors"])
df["persons"] = preprocessing.scale(df["persons"])
df["lug_boot"] = preprocessing.scale(df["lug_boot"])
df["safety"] = preprocessing.scale(df["safety"])

# writing pre-processed data to a csv file
df.to_csv('preProcessedCarDataset.csv', sep=',', index=False, header=False)
# split dataset in to train and test
train, test = train_test_split(df, test_size=split)
train.to_csv('train.csv', sep=',', index=False, header=False)
test.to_csv('test.csv', sep=',', index=False, header=False)