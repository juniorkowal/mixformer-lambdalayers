import torch

def train(train_loader, net, optimizer, criterion, device, label_smoothing, smoothing=False):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
        device: Device ('cuda' or 'cpu') the training is performed on.
        label_smoothing: Label smoothing component if smoothing is True.
        smoothing: Boolean, if True applies label smoothing.

    Returns:
        Average loss and accuracy for this training epoch.
    """
    # Added a net.train() cause it seems they forgot to add it?
    net.train()
    avg_loss = 0
    correct = 0
    total = 0

    # iterate through batches
    for i, data in enumerate(train_loader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Move data to target device
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        if smoothing:
            loss = label_smoothing(outputs, labels)
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total