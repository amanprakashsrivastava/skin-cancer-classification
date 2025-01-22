import torch
checkpoint = torch.load('models/model.pth', map_location='cpu')

# Print the checkpoint to inspect its structure
print(checkpoint)
