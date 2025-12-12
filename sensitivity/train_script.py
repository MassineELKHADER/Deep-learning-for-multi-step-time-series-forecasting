import torch
from loss.dilate_loss import dilate_loss

def train_model(
    trainloader,
    device,
    net,
    loss_type,
    learning_rate,
    alpha,
    gamma,
    epochs=150,
):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    net.train()
    for _ in range(epochs):
        for inputs, targets, _ in trainloader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            outputs = net(inputs)
            if (loss_type=='mse'):
                criterion = torch.nn.MSELoss()
                loss = criterion(outputs, targets)
            if (loss_type=='dilate'):
                loss, _, _ = dilate_loss(outputs=outputs, targets=targets, alpha=alpha, gamma=gamma,device=device
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return net
