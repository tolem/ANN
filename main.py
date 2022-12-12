import warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from collections import  Counter
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from itertools import chain

# from  string import ascii_letters, punctuation
matplotlib.use('TkAgg')

df = pd.read_csv('data/train.csv')
warnings.filterwarnings("ignore")

def plot_missig_cols():  # shows a heatmap with missing cols
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()


# print(df['Title'].shape)
print(df.columns)
# plot_missig_cols()
colname = df.columns
missing_col = [col for col in colname if df[col].isnull().sum() > 0]

# shows class distribution with boxpolot
# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass', y='Age', data=df, palette='winter')
# plt.show()


# function  uses the class distribution from boxplot to update misssing rows
def impute_age(colsname):
    age = colsname[0]
    pclass = colsname[1]
    if pd.isnull(age):
        if pclass == 1:
            return 37

        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)
# plot_missig_cols()


def cabin_check(columns: 'outputs a boolean expr for all cols'):
    cabin = columns[0]
    if pd.isnull(cabin):
        return 0
    else:
        return 1


# encoding Cabin cols to handle missing cols
df['Cabin'] = df[['Cabin']].apply(cabin_check, axis=1)

# X, y = df.drop(['Survived', 'Name'], axis=1),  df['Survived']
numerical = [col for col in df.columns if df[col].dtype != "O"]  # selects numerical cols
numerical.remove('Cabin')  # removes  cabin since its a nan
numerical.remove('Survived')  # removes survived since it's the target column

X = pd.concat([df[numerical], pd.get_dummies(df.Cabin), pd.get_dummies(df.Embarked),
                     pd.get_dummies(df.Sex), ], axis=1)
y = df.Survived
X.columns = X.columns.astype(str)
smote = SMOTE()
X.info()
y.info()
X_res, y_res = smote.fit_resample(X, y)
nearest_neigh_removed = Counter(y)[0] - Counter(y_res)[0]
print('Before oversampling with Smote %s' % (Counter(y)),
      'After oversampling with Smote %s'  % Counter(y_res),
      sep='\n'
      )

# creating   test  and  train set
X_train, X_test, y_train, y_test = \
    train_test_split(X_res,
                     y_res, test_size=0.2,
                     random_state=100)


cols = X_train.columns
# scaler instance
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  # fit train with scaler
X_test = scaler.fit_transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
# print(len(cols))
# print(X_test)
X_test = pd.DataFrame(X_test, columns=[cols])
# print(X_train.shape)
# exit()

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


batch_size = 64
# Instantiate training and test data
train_data = Data(X_train.to_numpy(), y_train.to_numpy())
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = Data(X_test.to_numpy(), y_test.to_numpy())
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch + 1}")
    print(f"X shape: {X.shape} ")
    print(f"y shape: {y.shape}")
    print('===================== Neural Networks =====================')


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1,  hidden_dim2, output_dim):
        super(NeuralNetwork, self).__init__()
        # Define the first hidden layer
        self.hidden1 = nn.Linear(in_features=input_dim, out_features=hidden_dim1)
        self.relu1 = nn.ReLU()

        # Define the second hidden layer
        self.hidden2 = nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2)
        self.tanh2 = nn.Tanh()

        self.output = nn.Linear(in_features=hidden_dim2, out_features=output_dim)
        self.relu2 = nn.Sigmoid()

        # init
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        # Compute the output of the first hidden layer
        x = self.hidden1(x)
        x = self.relu1(x)

        # Compute the output of the second hidden layer
        x = self.hidden2(x)
        x = self.tanh2(x)
        x = self.output(x)
        x = self.relu2(x)
        return x


# Instantiate the neural network
model = NeuralNetwork(13, 5, 3, 1)
print(model)
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()
num_epochs = 100
loss_values = []


for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
print("Training Complete")


print(len(loss_values), loss_values[:5])
step = np.linspace(0, 20, 1400)
fig, ax = plt.subplots(figsize=(20, 10))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# model.eval()
y_pred, y_gt = [], []
total, correct = 0, 0
with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        predicted = np.where(outputs > 0.4, 1, 0)
        predicted = list(chain(*predicted))
        y_pred.append(predicted)
        y_gt.append(y)
        total += y.size(0)
        correct += (predicted == y.numpy()).sum().item()
print(f'Accuracy of the network on the test instances: {100 * correct // total}%')


y_pred = list(chain(*y_pred))
y_gt = list(chain(*y_gt))
# report
print(classification_report(y_gt, y_pred))
