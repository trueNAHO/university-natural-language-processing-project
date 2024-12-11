import json
import sys
from PIL import Image
import torch
import torch.nn as nn
import clip

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the trained CLIP model
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Define the classifier architecture
class CLIPCrossProductClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),  # Flatten cross-product matrix
            nn.Linear(embedding_dim ** 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)  # Output a single logit
        )

    def forward(self, image_embeds, text_embeds):
        image_embeds = nn.functional.normalize(image_embeds, p=2, dim=1)
        text_embeds = nn.functional.normalize(text_embeds, p=2, dim=1)
        cross_product = torch.bmm(image_embeds.unsqueeze(2), text_embeds.unsqueeze(1))
        logits = self.fc(cross_product)
        return logits

# Load the trained model checkpoint
checkpoint_path = "../model_checkpoint/model_cross_product.pt"
print(f"Loading model checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)

embedding_dim = model.visual.output_dim
classifier = CLIPCrossProductClassifier(embedding_dim=embedding_dim).to(device)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()

# Function to predict if an image-text pair is harmful
def predict(image_path, text):
    # Preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Tokenize the text
    text_tokenized = clip.tokenize([text], context_length=77).to(device)
    
    # Get embeddings
    with torch.no_grad():
        image_embeds = model.encode_image(image)
        text_embeds = model.encode_text(text_tokenized)

        # Predict using the classifier
        logits = classifier(image_embeds, text_embeds).squeeze(1)
        probs = torch.sigmoid(logits).item()

    # Return the label based on threshold
    return "Harmful" if probs > 0.5 else "Not Harmful", probs

# Main script
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python use.py <image_path> <text>")
        sys.exit(1)

    image_path = sys.argv[1]
    text = sys.argv[2]

    # Perform prediction
    label, probability = predict(image_path, text)
    print(f"Prediction: {label} (Probability: {probability:.2f})")
