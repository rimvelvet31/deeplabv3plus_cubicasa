import os
import torch
import matplotlib.pyplot as plt
import customtkinter as ctk
from PIL import Image

class MetricsUI(ctk.CTkFrame):
    def __init__(self, root, tensor, model_output_path):
        super().__init__(root)
        self.metrics = tensor
        self.model_output = model_output_path  # Load the model output from the file

        self.save_maps()
        self.create_widgets()
        
        self.pack(expand=True, fill="both")

    def create_widgets(self):
        # Metrics display
        metrics_frame = ctk.CTkFrame(self)
        metrics_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(metrics_frame, text="Metrics:", font=("Arial", 16, "bold")).pack()
        
        # Dynamically add metric labels
        for metric, value in self.metrics:
            ctk.CTkLabel(metrics_frame, text=f"{metric}: {value}").pack()

        # Images display
        images_frame = ctk.CTkFrame(self)
        images_frame.pack(padx=10, pady=10, expand=True, fill="both")

        # Store images as instance attributes to prevent garbage collection
        self.seg_map_image = ctk.CTkImage(Image.open("maps/segmentation_map.png"), size=(300, 300))
        self.icon_map_image = ctk.CTkImage(Image.open("maps/icon_map.png"), size=(300, 300))

        ctk.CTkLabel(images_frame, text="Segmentation Map").grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(images_frame, image=self.seg_map_image).grid(row=1, column=0, padx=5, pady=5)

        ctk.CTkLabel(images_frame, text="Icon Map").grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(images_frame, image=self.icon_map_image).grid(row=1, column=1, padx=5, pady=5)

    def save_maps(self):
        # Create 'maps' folder if it doesn't exist
        if not os.path.exists("maps"):
            os.makedirs("maps")

        # Save segmentation map from model output
        plt.figure()
        plt.imshow(self.model_output[21].cpu().numpy(), cmap='viridis')  # Convert to numpy if it's a tensor
        plt.axis('off')
        plt.savefig("maps/segmentation_map.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save icon map from model output
        plt.figure()
        plt.imshow(self.model_output[22].cpu().numpy(), cmap='viridis')  # Convert to numpy if it's a tensor
        plt.axis('off')
        plt.savefig("maps/icon_map.png", bbox_inches='tight', pad_inches=0)
        plt.close()

# class Initializer():
#     def __init__(self, tensor, model_output_path):
#         MetricsUI(tensor, model_output_path).mainloop()

if __name__ == "__main__":
    # Example usage (you'll need to replace these with your actual tensor, model_output path, and metrics)
    import numpy as np
    tensor = np.random.rand(10, 10)
    model_output_path = 'path/to/model_output_file.pth'  # path to your torch file
    metrics = [["FIOU", 21], ["MIOU", 22]]
    
    app = MetricsUI(tensor, model_output_path, metrics)
    app.mainloop()