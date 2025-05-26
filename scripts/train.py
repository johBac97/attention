import torch
import torchvision
import tqdm
import datasets
import attention


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = sample["image"]
        label = sample["label"]

        if self.transforms:
            image = self.transforms(image)

        return image, label


def do_loop(model, dataloader, optimizer, loss_fn, device, train: bool = False):
    if train:
        model.train()
        desc_string = "Train Epoch"
    else:
        model.eval()
        desc_string = "Val Epoch"

    total_loss = 0
    number_correct = 0

    for images, labels in tqdm.tqdm(
        dataloader, total=len(dataloader), desc=desc_string
    ):
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()
            model_out = model(images)

            loss = loss_fn(model_out, labels)

            loss.backward()

            optimizer.step()
        else:
            with torch.no_grad():
                model_out = model(images)

                loss = loss_fn(model_out, labels)

        # The loss is averaged over the batch
        total_loss += loss.item() * images.size(0)
        number_correct += (model_out.argmax(dim=1) == labels).sum().item()

    acc = number_correct / len(dataloader.dataset)
    return acc, total_loss


def main():
    number_epochs = 20
    batch_size = 128
    embed_dim = 8
    patch_size = 4
    lr = 1e-3

    data = datasets.load_dataset("ylecun/mnist")

    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    val_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = MNISTDataset(data["train"], train_transforms)
    val_dataset = MNISTDataset(data["test"], val_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    model = attention.VisionTransformer(
        number_classes=10, embed_dim=embed_dim, patch_size=patch_size
    )

    number_parameters = sum([x.numel() for x in model.parameters()])
    print(f"Number of Parameters (thousands):\t{number_parameters / 1e3:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    optimizer = torch.optim.Adafactor(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch_idx in range(number_epochs):
        train_acc, train_loss = do_loop(
            model, train_dataloader, optimizer, loss_fn, device, train=True
        )

        val_acc, val_loss = do_loop(
            model, val_dataloader, optimizer, loss_fn, device, train=False
        )

        print(
            f"Epoch {epoch_idx + 1}: Train Loss={train_loss:.2f}, Train Accuracy={train_acc:.4f}, Val Loss={val_loss:.2f}, Val Accuracy={val_acc:.4}"
        )

    torch.save(model.state_dict(), "vit_model.pth")


if __name__ == "__main__":
    main()
