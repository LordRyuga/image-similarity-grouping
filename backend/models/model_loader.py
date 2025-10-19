import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load available models
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        models_dir = 'saved_models'
        
        # Check if SimCLR model exists
        simclr_path = os.path.join(models_dir, 'simclr_model.pth')
        if os.path.exists(simclr_path):
            try:
                # Load your trained SimCLR model
                # Adjust this based on your model architecture
                model = torch.load(simclr_path, map_location=self.device)
                model.eval()
                self.models['simclr'] = model
                print(f"Loaded SimCLR model from {simclr_path}")
            except Exception as e:
                print(f"Error loading SimCLR model: {e}")
                # Fallback to ResNet50 for demo purposes
                self._load_fallback_model()
        else:
            print("SimCLR model not found, using fallback ResNet50")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model (ResNet50) for demonstration"""
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer to get embeddings
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        self.models['resnet50'] = model
        print("Loaded fallback ResNet50 model")
    
    def get_available_models(self):
        """Return list of available model names"""
        return list(self.models.keys())
    
    def generate_embeddings(self, model_name, image_paths):
        """Generate embeddings for a list of images"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        embeddings = []
        
        with torch.no_grad():
            for image_path in image_paths:
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Generate embedding
                    embedding = model(image_tensor)
                    embedding = embedding.cpu().numpy().flatten()
                    embeddings.append(embedding)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    # Add zero embedding as placeholder
                    embeddings.append(np.zeros(2048))  # Adjust size based on your model
        
        return np.array(embeddings)
