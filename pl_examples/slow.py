import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, Dataset, DataLoader


l_rate = 0.2
mse_loss = nn.MSELoss(reduction = 'mean')

df = pd.read_csv('bike_sharing_hourly.csv')
print(df.head(5))

onehot_fields = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
for field in onehot_fields:
    dummies = pd.get_dummies(df[field], prefix=field, drop_first=False)
    df = pd.concat([df, dummies], axis=1)
df = df.drop(onehot_fields, axis = 1)
print(df.head(5))

continuous_fields = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for field in continuous_fields:
    mean, std = df[field].mean(), df[field].std()
    scaled_features[field] = [mean, std]
    df.loc[:, field] = (df[field] - mean)/std
scaled_features

df_backup = df.copy()

fields_to_drop = ['instant', 'dteday', 'atemp', 'workingday']
df.drop(fields_to_drop, axis=1, inplace = True)

# Split of 60 days of data from the end of the df for validation
validation_data = df[-60*24:]
df = df[:-60*24]

# Split of 21 days of data from the end of the df for testing
test_data = df[-21*24:]
df = df[:-21*24]

# The remaining (earlier) data will be used for training
train_data = df

# What have we ended up with?
print(f'''Validation data length: {len(validation_data)}
Test data length: {len(test_data)}
Train data length: {len(train_data)}''')

target_fields = ['cnt', 'casual', 'registered']

train_features, train_targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
validation_features, validation_targets = validation_data.drop(target_fields, axis=1), validation_data[target_fields]


class Regression(pl.LightningModule):

    ### The Model ###

    # Question: what will your model architecture look like?
    # Initialize the layers
    # Here we have one input layer (size 56 as we have 56 features), one hidden layer (size 10),
    # and one output layer (size 1 as we are predicting a single value)
    def __init__(self):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(56, 10)
        self.fc2 = nn.Linear(10, 1)

    # Question: how should the forward pass be performed, and what will its ouputs be?
    # Perform the forward pass
    # We're using the sigmoid activation function on our hidden layer, but our output layer has no activation
    # function as we're predicting a continuous variable so we want the actual number predicted
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

    ### The Data Loaders ###

    # Question: How do you want to load your data into the model?
    # Define functions for data loading: train / validate / test
    def train_dataloader(self):
        train_dataset = TensorDataset(torch.tensor(train_features.values).float(), torch.tensor(train_targets[['cnt']].values).float())
        train_loader = DataLoader(dataset = train_dataset, batch_size = 128)
        return train_loader

    def val_dataloader(self):
        validation_dataset = TensorDataset(torch.tensor(validation_features.values).float(), torch.tensor(validation_targets[['cnt']].values).float())
        validation_loader = DataLoader(dataset = validation_dataset, batch_size = 128)
        return validation_loader

    def test_dataloader(self):
        test_dataset = TensorDataset(torch.tensor(test_features.values).float(), torch.tensor(test_targets[['cnt']].values).float())
        test_loader = DataLoader(dataset = test_dataset, batch_size = 128)
        return test_loader

    ### The Optimizer ###

    # Question: what optimizer will I use?
    # Define optimizer function: here we are using Stochastic Gradient Descent
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=l_rate)

    ### Training ###

    # Question: what should a training step look like?
    # Define training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = mse_loss(logits, y)
        # Add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    ### Validation ###

    # Question: what should a validation step look like?
    # Define validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = mse_loss(logits, y)
        return {'val_loss': loss}

    # Define validation epoch end
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    ### Testing ###

    # Question: what should a test step look like?
    # Define test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = mse_loss(logits, y)
        correct = torch.sum(logits == y.data)

        # I want to visualize my predictions vs my actuals so here I'm going to
        # add these lines to extract the data for plotting later on
        predictions_pred.append(logits)
        predictions_actual.append(y.data)
        return {'test_loss': loss, 'test_correct': correct, 'logits': logits}

    # Define test end
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs }


predictions_pred = []
predictions_actual = []

model = Regression()
trainer = pl.Trainer(max_epochs=1000, gpus=2, num_nodes=1, distributed_backend="ddp")
trainer.fit(model) #, DataLoader(train), DataLoader(val))
#pl.trainer(max_epochs=1, gpus=8) #, num_nodes=32)) #.test()