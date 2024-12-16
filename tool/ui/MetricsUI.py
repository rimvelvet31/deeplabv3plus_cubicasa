import os
import torch

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
import numpy as np
from matplotlib.colors import ListedColormap

import customtkinter as ctk
from PIL import Image

from icecream import ic



class MetricsUI(ctk.CTkFrame):
    def __init__(self, root, metrics, model_to_use, COLOR_PRESETS):
        super().__init__(root)
        self.winfo_toplevel().title("Metrics")
        self.configure(bg_color=COLOR_PRESETS.BG_COLOR, fg_color=COLOR_PRESETS.BG_COLOR)

        self.metrics = metrics
        self.model_to_use = model_to_use

        self._createWidgets()
        self.pack(expand=True, fill="both")
    
    def _createWidgets(self):
        def _list_metrics(metric_list):
            class_pixel_accuracies = ctk.CTkTextbox(metrics_frame, font=("Arial", 12, "bold"),
                                                    width=350, height=200,
                                                    bg_color="transparent", fg_color="transparent",
                                                    activate_scrollbars=False,
                                                    wrap="word")
            class_pixel_accuracies.grid(row=2, column=0)
            metrics_computation = metric_list[-3:]
            metric_list = metric_list[:-3]
            for j, (metric, value) in enumerate(metric_list):
                class_pixel_accuracies.insert(str((j%12)+1)+".0" if j < 12 else str((j%12)+1)+".end", f"\u2022 {metric} = {value}\n" if j < 12 else f"\t\t\t\u2022 {metric} = {value}")
            class_pixel_accuracies.configure(state="disabled")
            for j, (metric, value) in enumerate(metrics_computation):
                ctk.CTkLabel(metrics_frame, text=f"{metric} = {value}", font=("Arial", 14, "bold")).grid(row=j+3, column=0, pady=5, sticky="w")

        for i, metric_list in enumerate(self.metrics):
            metrics_frame = ctk.CTkFrame(self)
            metrics_frame.configure(fg_color="transparent", bg_color="transparent")
            metrics_frame.pack(padx=10, pady=10, anchor="center", side="left", expand=True)

            if self.model_to_use < 2:
                ctk.CTkLabel(metrics_frame, text="DeepLabV3+" if self.model_to_use == 0 else "DeepLabV3+ w/ CA & SA", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
            else:
                ctk.CTkLabel(metrics_frame, text="DeepLabV3+" if i == 0 else "DeepLabV3+ w/ CA & SA", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
            ctk.CTkLabel(metrics_frame, text="Class Pixel Accuracy", font=("Arial", 14, "bold")).grid(row=1, column=0, sticky="w")
            _list_metrics(metric_list)

class SegmentationUI(ctk.CTkFrame):
    def __init__(self, root, model_output_path, model_to_use, COLOR_PRESETS):
        super().__init__(root)
        self.winfo_toplevel().title("Segmentation Maps")
        self.configure(bg_color=COLOR_PRESETS.BG_COLOR, fg_color=COLOR_PRESETS.BG_COLOR)

        self.size = (250, 250)
        self.output = model_output_path

        self.model_to_use = model_to_use


        self.room_classes = [
            "Background", "Outdoor", "Wall", "Kitchen", "Living Room",
            "Bed Room", "Bath", "Entry", "Railing", "Storage", 
            "Garage", "Undefined"
        ]
        self.icon_classes = [
            "No Icon", "Window", "Door", "Closet", "Electrical Appliance",
            "Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", 
            "Chimney"
        ]

        self.room_colors = plt.cm.tab20(np.linspace(0, 1, len(self.room_classes)))
        self.icon_colors = plt.cm.tab20b(np.linspace(0, 1, len(self.icon_classes)))

        self.room_cmap = ListedColormap(self.room_colors)
        self.icon_cmap = ListedColormap(self.icon_colors)

        self._create_legends()

        self._create_widgets()

        self.pack(expand=True, fill="both", anchor="center")

    def _create_legends(self):
        def _save_legend(class_names, colors, output_path, title):
            patches = [Patch(color=colors[i], label=class_name) for i, class_name in enumerate(class_names)]

            rcParams.update({'text.color': 'white', 'axes.edgecolor': 'white'})

            plt.figure(figsize=(2, len(class_names) * 0.4))
            legend = plt.legend(
                handles=patches, 
                loc='center', 
                title=title, 
                title_fontsize=12, 
                fontsize=16, 
                frameon=False,
                handlelength=1,
                handletextpad=0.5,
                borderpad=0
            )
            
            legend.get_title().set_color("white")
            for text in legend.get_texts():
                text.set_color("white")
            
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=150)
            plt.close()

        _save_legend(self.room_classes, self.room_colors, "maps/room_legend.png", "Room Classes")
        _save_legend(self.icon_classes, self.icon_colors, "maps/icon_legend.png", "Icon Classes")

    def _create_widgets(self):
        def _save_maps(output):
            if not os.path.exists("maps"):
                os.makedirs("maps")

            plt.figure()
            plt.imshow(output[0].cpu().numpy(), cmap=self.room_cmap, vmin=0, vmax=len(self.room_classes)-1)
            plt.axis('off')
            plt.savefig("maps/segmentation_map.png", bbox_inches='tight', pad_inches=0)
            plt.close()

            # Icon segmentation map
            plt.figure()
            plt.imshow(output[1].cpu().numpy(), cmap=self.icon_cmap, vmin=0, vmax=len(self.icon_classes)-1)
            plt.axis('off')
            plt.savefig("maps/icon_map.png", bbox_inches='tight', pad_inches=0)
            plt.close()

        for i, output in enumerate(self.output):
            images_frame = ctk.CTkFrame(self)
            images_frame.configure(fg_color="transparent", bg_color="transparent")
            images_frame.pack(padx=10, pady=10, anchor="center", side="left", expand=True)
            _save_maps(output)

            if self.model_to_use == 0 and i == 0:
                ctk.CTkLabel(images_frame, text="DeepLabV3+", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
            elif self.model_to_use == 1 and i == 0:
                ctk.CTkLabel(images_frame, text="DeepLabV3+ w/ CA & SA", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
            elif self.model_to_use == 2 and i < 2:
                ctk.CTkLabel(images_frame, text="DeepLabV3+" if i == 0 else "DeepLabV3+ w/ CA & SA", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
            elif (self.model_to_use < 2 and i == 1) or (self.model_to_use == 2 and i == 2):
                ctk.CTkLabel(images_frame, text="Ground Truth", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

            self.seg_map_image = ctk.CTkImage(Image.open("maps/segmentation_map.png"), size=self.size)
            self.icon_map_image = ctk.CTkImage(Image.open("maps/icon_map.png"), size=self.size)

            ctk.CTkLabel(images_frame, text="Segmentation Map", font=("Arial", 12, "bold")).grid(row=1, column=0, padx=5, pady=5)
            ctk.CTkLabel(images_frame, image=self.seg_map_image, text="").grid(row=2, column=0, padx=5, pady=5)

            ctk.CTkLabel(images_frame, text="Icon Map", font=("Arial", 12, "bold")).grid(row=3, column=0, padx=5, pady=5)
            ctk.CTkLabel(images_frame, image=self.icon_map_image, text="").grid(row=4, column=0, padx=5, pady=5)

        legends_frame = ctk.CTkFrame(self)
        legends_frame.configure(fg_color="transparent", bg_color="transparent")
        legends_frame.pack(padx=10, pady=10, anchor="center", side="right", expand=True)

        room_legend_image = ctk.CTkImage(Image.open("maps/room_legend.png"), size=(100,250))
        icon_legend_image = ctk.CTkImage(Image.open("maps/icon_legend.png"), size=(100,250))

        ctk.CTkLabel(legends_frame, text="Room Classes Legend", font=("Arial", 12, "bold")).pack(padx=5, pady=5)
        ctk.CTkLabel(legends_frame, image=room_legend_image, text="").pack(padx=5, pady=5)

        ctk.CTkLabel(legends_frame, text="Icon Classes Legend", font=("Arial", 12, "bold")).pack(padx=5, pady=5)
        ctk.CTkLabel(legends_frame, image=icon_legend_image, text="").pack(padx=5, pady=5)
