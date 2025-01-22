import torch

# Load the entire model (including architecture and weights)
checkpoint = torch.load('models/model.pth', map_location='cpu')
classification_model = checkpoint  # Directly assign the loaded model

# Alternatively, if it's a weights-only checkpoint:
classification_model.load_state_dict(checkpoint)
