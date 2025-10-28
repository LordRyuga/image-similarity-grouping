image_tensor = self.transform(image)
image_tensor = image_tensor.unsqueeze(0).to(self.device)