import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

BASE_PATH_DATA = '/Users/magnusson/PycharmProjects/ml-cloud-opt-thick/data/skogsstyrelsen'

epochs = 100
gamma = 1e-5  # Learning rate
gamma_loss = 0.95
H1 = 100
batch_size = 100

# Get the data
img_paths_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_train.npy')))
img_paths_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_val.npy')))
img_paths_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_test.npy')))

# Labels
json_content_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_train.npy'), allow_pickle=True))
json_content_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_val.npy'), allow_pickle=True))
json_content_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_test.npy'), allow_pickle=True))

# Network
network = nn.Sequential(
    nn.Linear(20 * 20, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 2),
    nn.Sigmoid()  # Sigmoid activation for binary classification
)

loss_function = torch.nn.MSELoss()  # What function to use to calculate the loss given the prediction and labels

optimizer = torch.optim.SGD(network.parameters(),
                            lr=1)  # Function for updating the parameters of the network based on loss

BAND_NAMES = ['b01', 'b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b8a', 'b09', 'b11', 'b12']

train_losses = []
val_losses = []

# Load the model

## Set models learning rate


DO_TRAINING = False

BEST_LOSS = 1000

if DO_TRAINING:

    for epoch in range(epochs):

        network.eval()
        # Validation
        for img_idx, img_path in enumerate(img_paths_val):
            img_path = img_path.replace("../data", "/Users/magnusson/PycharmProjects/ml-cloud-opt-thick/data")
            img = xr.open_dataset(img_path)

            band_list = []
            for band_name in BAND_NAMES:
                band_list.append((getattr(img, band_name).values - 1000) / 10000)
            img = np.concatenate(band_list, axis=0)
            img = np.transpose(img, [1, 2, 0])
            img = img[:20, :20, :]
            img_tensor = torch.tensor(img, dtype=torch.float32)
            img_tensor = img_tensor.reshape(1, -1, 20 * 20)

            molndis = json_content_val[img_idx]['MolnDis']
            molndis_tensor = torch.tensor([int(molndis)], dtype=torch.float32).repeat(1, 12, 2)

            prediction = network(img_tensor)
            loss = loss_function(prediction, molndis_tensor)
            val_losses.append(loss.item())
            print(f'\rEpoch {epoch + 1}/{epochs}, img {img_idx + 1}/{len(img_paths_val)}, loss {loss}', end='')
            # Save the model if it's the best one
            if loss < BEST_LOSS:
                if loss == 0:
                    continue
                print(f'\nNew best loss: {loss}')
                BEST_LOSS = loss
                torch.save(network.state_dict(), 'model.pth')

        network.train()
        # Training
        for img_idx, img_path in enumerate(img_paths_train):

            # Read the image
            img_path = img_path.replace("../data", "/Users/magnusson/PycharmProjects/ml-cloud-opt-thick/data")
            img = xr.open_dataset(img_path)

            band_list = []
            for band_name in BAND_NAMES:
                band_list.append((getattr(img, band_name).values - 1000) / 10000)  # -1k and then 10k division
            img = np.concatenate(band_list, axis=0)
            img = np.transpose(img, [1, 2, 0])
            img = img[:20, :20, :]

            # Convert NumPy array to PyTorch tensor
            img_tensor = torch.tensor(img, dtype=torch.float32)
            img_tensor = img_tensor.reshape(1, -1, 20 * 20)

            # Ready the label
            molndis = json_content_train[img_idx]['MolnDis']
            molndis_tensor = torch.tensor([int(molndis)], dtype=torch.float32).repeat(1, 12, 2)
            # print(molndis_tensor)

            prediction = network(img_tensor)

            # print(prediction)

            loss = loss_function(prediction, molndis_tensor)

            # Backpropogate the loss through the network to find the gradients of all parameters
            loss.backward()

            # Update the parameters along their gradients
            optimizer.step()

            # Clear stored gradient values
            optimizer.zero_grad()

            train_losses.append(loss.item())

            print(f'\rEpoch {epoch + 1}/{epochs}, img {img_idx + 1}/{len(img_paths_train)}, loss {loss}', end='')

# Plot the training loss per epoch
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.show()

plt.plot(val_losses)
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.show()

# Check if the model exists
if os.path.exists('model.pth'):
    network.load_state_dict(torch.load('model.pth'))

network.eval()
accuracy = 0
total = 0
# Test
for img_idx, img_path in enumerate(img_paths_test):
    img_path = img_path.replace("../data", "/Users/magnusson/PycharmProjects/ml-cloud-opt-thick/data")
    img = xr.open_dataset(img_path)

    band_list = []
    for band_name in BAND_NAMES:
        band_list.append((getattr(img, band_name).values - 1000) / 10000)
    img = np.concatenate(band_list, axis=0)
    img = np.transpose(img, [1, 2, 0])
    img = img[:20, :20, :]
    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = img_tensor.reshape(1, -1, 20 * 20)

    molndis = json_content_test[img_idx]['MolnDis']

    with torch.no_grad():
        output = network(img_tensor)
    # prediction = network(img_tensor)
    # Get Binary prediction
    # prediction = torch.argmax(prediction, dim=0)

    # if any value is above 0.5, then it's a cloud

    # print(output.detach().numpy())


    # print(f'Prediction: {predicted} | True: {molndis}')
    # print(output.argmax(dim=1))
    #print(output.mean().detach().numpy() > 0.5)
    #print(molndis)
    #print(f'{output.sum()} | True: {molndis}')

    if output.mean().detach().numpy() >= 0.5 and int(molndis) == 1:
        accuracy += 1
    elif output.mean().detach().numpy() < 0.5 and int(molndis) == 0:
        accuracy += 1

    #if output.sum() > 2 and int(molndis) == 1:
    #    accuracy += 1
    #elif output.sum() < 2 and int(molndis) == 0:
    #    accuracy += 1
    total += 1

print(f'Accuracy: {accuracy / total * 100}%')

# print(f'\rimg {img_idx + 1}/{len(img_paths_test)}, loss {loss}', end='')