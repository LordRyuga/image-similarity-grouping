import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import cast
import os
from models.simclr_model import ResNetSimCLR

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
        """Load all available models from saved_models directory"""
        models_dir = 'saved_models'
        
        # 1. Try to load SimCLR model
        self._load_simclr_model(models_dir)
        
        # 2. Try to load DCL model
        self._load_dcl_model(models_dir)
        
        # 3. Always load ResNet50 as fallback
        if len(self.models) == 0:
            print("⚠️ No custom models found, loading fallback ResNet50")
            self._load_fallback_model()
        else:
            print(f"✅ Loaded {len(self.models)} model(s): {list(self.models.keys())}")
    
    def _load_simclr_model(self, models_dir):
        """Load SimCLR model with 512-dim embeddings"""
        simclr_path = os.path.join(models_dir, 'simclr_model.pth')
        
        if os.path.exists(simclr_path):
            try:
                # SimCLR with 512-dim output
                model = ResNetSimCLR(base_model="resnet18", out_dim=512)
                checkpoint = torch.load(simclr_path, map_location=self.device)

                # Handle checkpoint dictionary
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    epoch = checkpoint.get('epoch', 'unknown')
                    print(f"📊 SimCLR checkpoint from epoch {epoch}")
                else:
                    state_dict = checkpoint

                # Remove "module." prefix (DataParallel wrapper)
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace("module.", "")] = v

                model.load_state_dict(new_state_dict, strict=False)
                model.eval()
                model.to(self.device)

                self.models['simclr'] = model
                print(f"✅ Loaded SimCLR model (feature_dim=512)")

            except Exception as e:
                print(f"❌ Error loading SimCLR model: {e}")
                import traceback
                traceback.print_exc()
    
    def _load_dcl_model(self, models_dir):
        """Load DCL model with 128-dim embeddings (CIFAR-10 trained)"""
        dcl_path = os.path.join(models_dir, 'dcl_model.pth')  # Different filename
        
        if os.path.exists(dcl_path):
            try:
                # DCL with 128-dim output (from your training config)
                model = ResNetSimCLR(base_model="resnet18", out_dim=128)
                checkpoint = torch.load(dcl_path, map_location=self.device)

                # Handle checkpoint dictionary
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    epoch = checkpoint.get('epoch', 'unknown')
                    print(f"📊 DCL checkpoint from epoch {epoch}")
                else:
                    state_dict = checkpoint

                # Remove "module." prefix (DataParallel wrapper)
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace("module.", "")] = v

                model.load_state_dict(new_state_dict, strict=False)
                model.eval()
                model.to(self.device)

                self.models['dcl'] = model
                print(f"✅ Loaded DCL model (CIFAR-10, feature_dim=128)")

            except Exception as e:
                print(f"❌ Error loading DCL model: {e}")
                import traceback
                traceback.print_exc()
    
    def _load_fallback_model(self):
        """Load fallback ResNet50 pretrained on ImageNet"""
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer to get embeddings
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        self.models['resnet50'] = model
        print("✅ Loaded fallback ResNet50 model")
    
    def get_available_models(self):
        """Return list of available model names"""
        return list(self.models.keys())
    
    def _get_embedding_dim(self, model_name):
        """Get the embedding dimension for each model"""
        embedding_dims = {
            'simclr': 512,
            'dcl': 128,
            'resnet50': 2048
        }
        return embedding_dims.get(model_name, 2048)
    
    def generate_embeddings(self, model_name, image_paths):
        """Generate embeddings for a list of images"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        embedding_dim = self._get_embedding_dim(model_name)
        embeddings = []
        
        with torch.no_grad():
            for image_path in image_paths:
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = cast(torch.Tensor, self.transform(image)).unsqueeze(0).to(self.device)

                    # Generate embedding
                    embedding = model(image_tensor)
                    embedding = embedding.cpu().numpy().flatten()
                    embeddings.append(embedding)

                except Exception as e:
                    print(f"❌ Error processing {image_path}: {e}")
                    # Add zero embedding as placeholder with correct dimension
                    embeddings.append(np.zeros(embedding_dim))
        
        return np.array(embeddings)