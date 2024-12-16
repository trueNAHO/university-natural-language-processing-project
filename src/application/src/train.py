import PIL.Image
import clip
import json
import torch
import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load training data
with open("../assets/train.jsonl", "r") as f:
    input_data = []
    img_paths = []
    text_data = []
    labels = []
    path = "../assets/"
    for line in f:
        obj = json.loads(line)
        input_data.append(obj)
        img_paths.append(path + obj["img"])
        text_data.append(obj["text"])
        labels.append(obj["label"])

# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
for param in clip_model.parameters():
    param.requires_grad = False


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_image_path, list_text, labels):
        self.image_path = list_image_path
        self.text = list_text
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.text)

    def __getitem__(self, id):
        image = self.preprocess(PIL.Image.open(self.image_path[id]))
        text = clip.tokenize(self.text[id], context_length=77, truncate=True).squeeze(0)
        label = self.labels[id]
        return image, text, label


dataset = Dataset(img_paths, text_data, labels)
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=6
)


class CLIPCrossProductClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(embedding_dim**2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, image_embeds, text_embeds):
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)

        cross_product = torch.bmm(image_embeds.unsqueeze(2), text_embeds.unsqueeze(1))
        logits = self.fc(cross_product)
        return logits


embedding_dim = clip_model.visual.output_dim
classifier = CLIPCrossProductClassifier(embedding_dim=embedding_dim).to(device)

# Loss function and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Training loop
num_epochs = 20
classifier.train()
for epoch in range(num_epochs):
    pbar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        images, texts, labels = batch

        images = images.to(device)
        texts = texts.to(device)
        labels = labels.clone().detach().to(dtype=torch.float32, device=device)

        with torch.no_grad():
            image_embeds = clip_model.encode_image(images)
            text_embeds = clip_model.encode_text(texts)

        logits = classifier(image_embeds, text_embeds).squeeze(1)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


def save():
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        "../model_checkpoint/model_cross_product.pt",
    )


saving = True
if saving:
    save()
