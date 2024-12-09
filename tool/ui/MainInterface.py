import os
import torch

from customtkinter import *
from tkinter import filedialog
from PIL import Image

from icecream import ic

from utils import load_model, load_dataset, load_img_and_labels, evaluate


class MainInterface():
    def __init__(self, root, PATHS, RECONSTUCTION):
        self.COLOR_PRESETS = PATHS.COLOR_PRESETS
        self.IMAGE_PATHS = PATHS.IMAGE_PATHS
        
        self.Renderer = RECONSTUCTION.Renderer()
        self.Vectorizer = RECONSTUCTION.Vectorizer()
        
        self.selected_floorplan = None
        self.model_to_use = None
        # self.run_both_models = False
        self.show_core_elements_only = False

        self.output = []
        self.labels = []

        self.root = root

        red = 'C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/data/cubicasa5k/'
        me = 'D:\GitHub\deepl_lab\data\cubicasa5k\\'

        self.dataset = load_dataset(me)
        
        main_frame = self._initializeMainFrame(root)
        self._loadContents(main_frame);
    
    def _initializeMainFrame(self, root):
        main_frame = CTkFrame(root)
        main_frame.place(relwidth=1, relheight=1)
        main_frame.configure(bg_color=self.COLOR_PRESETS.BG_COLOR, fg_color=self.COLOR_PRESETS.BG_COLOR)
        
        return main_frame

    def _loadContents(self, main_frame):
        upload_photo = CTkButton(main_frame)
        upload_photo.configure(corner_radius=32,
                               image=self._createCtkImage(self.IMAGE_PATHS.CAMERA_ICON, size=(20, 20)),
                               hover_color = self.COLOR_PRESETS.BTN_HOVER_COLOR,
                               fg_color="transparent", bg_color="transparent",
                               border_color="white", border_width=1,
                               text="Upload Photo", text_color="white",
                               command=lambda: self._setParameter("floorplan", self._uploadPhoto(upload_photo)))
        upload_photo.place(relx=0.28, rely=0.4, relwidth=0.4, relheight=0.7, anchor="center")
        
        generate_model_btn = CTkButton(main_frame, text="Generate 3D Model", text_color="black", font=CTkFont("Arial", 12, weight="bold"))
        generate_model_btn.configure(corner_radius=32,
                                     image=self._createCtkImage(self.IMAGE_PATHS.GENERATE_MODEL_ICON, size=(12, 16)),
                                     command=self._generateModel)
        generate_model_btn.place(relx=0.28, rely=0.8, relwidth=0.4, anchor="center")
        
        show_segmentation_maps_btn = CTkButton(main_frame, text="Show Segmentation Maps", text_color="black", font=CTkFont("Arial", 12, weight="bold"))
        show_segmentation_maps_btn.configure(corner_radius=32,
                                            image=self._createCtkImage(self.IMAGE_PATHS.MAP_ICON, size=(12, 12)),
                                            command=self._seeSegMaps)
        show_segmentation_maps_btn.place(relx=0.28, rely=0.865, relwidth=0.4, anchor="center")
        
        model_renderer_frame = CTkButton(main_frame)
        model_renderer_frame.configure(corner_radius=32,
                                       hover_color=self.COLOR_PRESETS.BTN_HOVER_COLOR,
                                       fg_color="transparent", bg_color="transparent",
                                       border_color="white", border_width=1,
                                       text="Show 3D Model", text_color="white",
                                       command=self._openRenderer)
        model_renderer_frame.place(relx=0.73, rely=0.4, relwidth=0.4, relheight=0.7, anchor="center")
        
        models_available = ["DeepLabv3+", "DeepLabv3+ (w/ CA & SA)", "Run Both Models"]
        select_model = CTkOptionMenu(main_frame, values=models_available)
        select_model.configure(corner_radius=32,
                               text_color="black", font=CTkFont("Arial", 12, weight="bold"), anchor="center",
                               dropdown_text_color="black", dropdown_font=CTkFont("Arial", 12, weight="bold"),
                               dropdown_fg_color="#1f6aa5", dropdown_hover_color=self.COLOR_PRESETS.BTN_HOVER_COLOR,
                               command=lambda selected_model: self._setParameter("model", models_available.index(selected_model)))
        select_model.place(relx=0.73, rely=0.8, relwidth=0.4, anchor="center")

        see_details_btn = CTkButton(main_frame, 
                                    text="See Details", 
                                    text_color="black", 
                                    font=CTkFont("Arial", 12, weight="bold"),
                                    image=self._createCtkImage(self.IMAGE_PATHS.SEE_DETAILS_ICON, size=(12, 12)),
                                    corner_radius=32,
                                    command=self._seeDetails)
        see_details_btn.place(relx=0.73, rely=0.865, relwidth=0.4, anchor="center")
        
        show_core_elements_only = CTkCheckBox(main_frame,
                                              text="Show only core floor plan elements")
        show_core_elements_only.configure(command=lambda: [self._setParameter("showCoreElementsOnly", show_core_elements_only.get())])
        show_core_elements_only.place(relx=0.5, rely=0.95, anchor="center")
    
    def _createCtkImage(self, image_path, size):
        return CTkImage(Image.open(image_path), size=size)
    
    def _uploadPhoto(self, button_widget):
        def pop_file(filepath):
            path, _ = os.path.split(filepath)
            return path
        
        selected_floorplan = filedialog.askopenfilename(title="Select a floorplan image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        
        path = pop_file(selected_floorplan) + "/"
        
        if selected_floorplan == "":
            return
        
        uploaded_image = Image.open(selected_floorplan)
        
        button_width = button_widget.winfo_width() - 65
        width, height = uploaded_image.size
        new_height = int((height * button_width) / width)

        ctk_image = CTkImage(light_image=uploaded_image, dark_image=uploaded_image, size=(button_width, new_height))
        
        button_widget.configure(image=ctk_image, text="")
        
        # return selected_floorplan
        return path
    
    def _setParameter(self, parameter, value):
        if value == None:
            print(f"Selected {parameter}: Not changed")
            return
        
        if parameter == "floorplan" and value != None:
            self.selected_floorplan = value
        elif parameter == "useBothModels":
            self.run_both_models = value
        elif parameter == "model":
            self.model_to_use = value
        
        print(f"Selected {parameter}: {value}")
    
    def _seeSegMaps(self):
        model_output_path = r"D:\GitHub\deepl_lab\tool\deeplab\floorplan_pred512.pt"
        model_output_path1 = r"D:\GitHub\deepl_lab\tool\deeplab\floorplan_pred512.pt"
        
        model_outputs = [model_output_path, model_output_path1]
        # ic(torch.load(model_output_path))
        self.root.seeSegMaps(model_outputs)
        # self.root.seeDetails(self.output, self.labels)
        
    def _seeDetails(self):
        metrics =[
                    ["Background", "0.98"],
                    ["Outdoor", "0.96"],
                    ["Wall", "0.94"],
                    ["Kitchen", "0.97"],
                    ["Living room", "0.92"],
                    ["Bedroom", "0.94"],
                    ["Bath", "0.95"],
                    ["Hallway", "0.99"],
                    ["Railing", "0.92"],
                    ["Storage", "0.90"],
                    ["Garage", "0.97"],
                    ["Other rooms", "0.92"],
                    ["Empty", "0.99"],
                    ["Window", "0.98"],
                    ["Door", "0.95"],
                    ["Closet", "0.99"],
                    ["Electr. Appl.", "0.97"],
                    ["Toilet", "0.90"],
                    ["Sink", "0.96"],
                    ["Sauna bench", "0.90"],
                    ["Fire Place", "0.93"],
                    ["Bathtub", "0.94"],
                    ["Chimney", "0.94"],
                    ["Other", "0.90"],
                    ["Mean Pixel Accuracy", "0.97"],
                    ["Mean IoU", "0.99"],
                    ["Frequency-Weighted IoU", "0.98"]
                ]
        metrics1 =[
                    ["Background", "0.98"],
                    ["Outdoor", "0.96"],
                    ["Wall", "0.94"],
                    ["Kitchen", "0.97"],
                    ["Living room", "0.92"],
                    ["Bedroom", "0.94"],
                    ["Bath", "0.95"],
                    ["Hallway", "0.99"],
                    ["Railing", "0.92"],
                    ["Storage", "0.90"],
                    ["Garage", "0.97"],
                    ["Other rooms", "0.92"],
                    ["Empty", "0.99"],
                    ["Window", "0.98"],
                    ["Door", "0.95"],
                    ["Closet", "0.99"],
                    ["Electr. Appl.", "0.97"],
                    ["Toilet", "0.90"],
                    ["Sink", "0.96"],
                    ["Sauna bench", "0.90"],
                    ["Fire Place", "0.93"],
                    ["Bathtub", "0.94"],
                    ["Chimney", "0.94"],
                    ["Other", "0.90"],
                    ["Mean Pixel Accuracy", "0.97"],
                    ["Mean IoU", "0.99"],
                    ["Frequency-Weighted IoU", "0.98"]
                ]
        metric_outputs = [metrics, metrics1]
        self.root.seeDetails(metric_outputs)

    def _generateModel(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img, labels = load_img_and_labels(self.dataset, self.selected_floorplan)
        
        models = []
        
        metrics = []
        outputs = []

        if self.model_to_use < 2:
            model = load_model(
                f"C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/tool/deeplab/best_checkpoint_mobilenetv2_{'base' if self.model_to_use == 0 else 'ca_sa'}.pt", 
                use_attention=False if self.model_to_use == 0 else True, device=device)
            combined_tensor, metrics_output = evaluate(model, img, labels)
            metrics.append(combined_tensor)
            outputs.append(metrics_output)
        else:
            for i in range(2):
                model = load_model(
                    f"C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/tool/deeplab/best_checkpoint_mobilenetv2_{'base' if i == 0 else 'ca_sa'}.pt", 
                    use_attention=False if i == 0 else True, device=device)
                models.append(model)
                combined_tensor, metrics_output = evaluate(model, img, labels)
                metrics.append(combined_tensor)
                outputs.append(metrics_output)
                print(f"Model loaded: deeplabv3plus_{model.backbone_name}_{model.attention}")

        
        self.labels = metrics
        self.output = outputs

        # self.Vectorizer(combined_tensor)
        # sample = torch.load(r"C:\Users\Red\Documents\GitHub\deeplabv3plus_cubicasa\tool\deeplab\floorplan_pred256.pt")
        scaled_rooms, scaled_outer_contour, scaled_walls, scaled_icons, icon_quadrilaterals, room_classes = self.Vectorizer.process_data(combined_tensor)
        self.Renderer.generate_model(scaled_rooms, scaled_outer_contour, scaled_walls, scaled_icons, icon_quadrilaterals, room_classes)
        print("Model generated")

    def _openRenderer(self):
        self.Renderer.show_model()