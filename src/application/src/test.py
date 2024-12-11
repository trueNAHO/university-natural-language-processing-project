import json
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, f1_score, precision_score, recall_score
from torchmetrics.classification import BinaryAUROC

import clip

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load test data
with open("../assets/dev.jsonl", "r") as f:
    input_data = []
    img_paths = []
    text_paths = []
    labels = []
    path = "../assets/"
    for line in f:
        obj = json.loads(line)
        input_data.append(obj)
        img_paths.append(path + obj["img"])
        text_paths.append(obj["text"])
        labels.append(obj["label"])

checkpoint = torch.load("../model_checkpoint/model_cross_product.pt", map_location=device)

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

class CLIPCrossProductClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim ** 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image_embeds, text_embeds):
        image_embeds = nn.functional.normalize(image_embeds, p=2, dim=1)
        text_embeds = nn.functional.normalize(text_embeds, p=2, dim=1)
        cross_product = torch.bmm(image_embeds.unsqueeze(2), text_embeds.unsqueeze(1))
        logits = self.fc(cross_product)
        return logits

embedding_dim = model.visual.output_dim
classifier = CLIPCrossProductClassifier(embedding_dim=embedding_dim).to(device)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()

class Dataset():
    def __init__(self, list_image_path, list_text):
        self.image_path = list_image_path
        self.text = clip.tokenize(list_text, context_length=77, truncate=True)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))
        text = self.text[idx]
        return image, text


test_dataset = Dataset(img_paths, text_paths)
test_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=8)
auroc_metric = BinaryAUROC().to(device)
all_predictions = []
all_labels = []

#evaluate model
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
        images, texts = batch
        images = images.to(device)
        texts = texts.to(device)

        image_embeds = model.encode_image(images)
        text_embeds = model.encode_text(texts)

        logits = classifier(image_embeds, text_embeds).squeeze(1)
        probs = torch.sigmoid(logits)

        predictions = (probs > 0.5).long()
        all_predictions.extend(predictions.cpu().numpy())

all_labels = torch.tensor(labels).numpy()
all_labels = torch.tensor(all_labels)
all_predictions = torch.tensor(all_predictions)


sklearn_auroc_score = roc_auc_score(all_labels.numpy(), all_predictions.numpy())

print(f"\nAUROC (Sklearn): {sklearn_auroc_score:.4f}\n")

# Compute evaluation variables
accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions, average='binary')
precision = precision_score(all_labels, all_predictions, average='binary')
recall = recall_score(all_labels, all_predictions, average='binary')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=["Not Harmful", "Harmful"]))
