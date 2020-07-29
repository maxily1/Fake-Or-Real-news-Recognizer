# Fake or Real news 

# Impoting libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing datasets in CSV format
real_data = pd.read_csv('Datasets/True.csv')
fake_data = pd.read_csv('Datasets/Fake.csv') 

# Printing the first few
print(real_data.head())
print(fake_data.head())

# Checking the length of each dataset
print(len(real_data))
print(len(fake_data))

# Editing the datasets
nb_articles = min(len(real_data), len(fake_data))
real_data = real_data[:nb_articles]
fake_data = fake_data[:nb_articles]

# Checking the length of each dataset after the edit
print(len(real_data))
print(len(fake_data))

# Adding new column
real_data['is_fake'] = False
fake_data['is_fake'] = True

# Checking the dataset after the edit
print(real_data.head())
print(fake_data.head())

# Importing another library
from sklearn.utils import shuffle

# Concatinating the real and fake data
data = pd.concat([real_data, fake_data])

# Shuffle the data
data = shuffle(data).reset_index(drop=True)
data.head()

# Now the fun part #
# Splitting the datasets to train_data, validate_data, test_data
train_data, validate_data, test_data = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])

# Resetting Index
train_data = train_data.reset_index(drop=True)
validate_data = validate_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Deleting the previous dataset
del real_data
del fake_data

# Printing size of sets
print("Size of training set: {}".format(len(train_data)))
print("Size of validation set: {}".format(len(validate_data)))
print("Size of testing set: {}".format(len(test_data)))

# Importing even more libraries
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Setting up a "device"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Setting up tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.config.num_labels = 1

# Freeze the pre trained parameters
for param in model.parameters():
    param.requires_grad = False

# Add three new layers at the end of the network
model.classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
    nn.Softmax(dim=1)
)

model = model.to(device)