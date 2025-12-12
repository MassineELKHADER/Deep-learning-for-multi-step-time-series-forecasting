import torch
from loss.dilate_loss import dilate_loss
from tqdm import tqdm
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
    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0)

    for epoch in epoch_bar:
        batch_bar = tqdm(
            trainloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            position=1,
        )
        running_loss = 0.0

        for inputs, targets, _ in batch_bar:
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
            running_loss += loss.item()
            batch_bar.set_postfix(loss=loss.item())

        epoch_bar.set_postfix(avg_loss=running_loss / len(trainloader))


    return net
