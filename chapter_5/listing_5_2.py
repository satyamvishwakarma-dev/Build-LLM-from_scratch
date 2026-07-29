from chapter_5.topic_5_1 import model
from chapter_5.listing_5_1 import train_loader, val_loader
import torch

from chapter_5.listing_5_1 import calc_loss_batch


def calc_loss_loader(data_loader, model, devices, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, devices
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)


if __name__ == "__main__":
    print(f"Training loss: {train_loss}")
    print(f"Validation loss: {val_loss}")
