import os
import threading
import torch

from customtkinter import *
from tkinter import filedialog
from PIL import Image

from icecream import ic

from utils import load_model, load_dataset, load_img_and_labels, evaluate

from floortrans import post_prosessing


class MainInterface():
    def __init__(self, root, PATHS, RECONSTRUCTION):
        self.COLOR_PRESETS = PATHS.COLOR_PRESETS
        self.IMAGE_PATHS = PATHS.IMAGE_PATHS
        self.RECONSTRUCTION = RECONSTRUCTION
        
        self.selected_floorplan = None
        self.model_to_use = 0
        self.show_core_elements_only = False

        self.output = []
        self.labels = []

        self.processed_data = []

        self.root = root

        # red = r'C:\Users\Red\Documents\GitHub\deeplabv3plus_cubicasa\data\cubicasa5k\\'
        # me = 'D:\GitHub\deepl_lab\data\cubicasa5k\\'

        DATASET_PATH = os.path.join(os.getcwd(), "data", "cubicasa5k/")
        
        self.dataset = load_dataset(DATASET_PATH)
        
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
        
        self._splitRenderer(main_frame)
        
        models_available = ["DeepLabv3+", "DeepLabv3+ (w/ CA & SA)", "Run Both Models"]
        select_model = CTkOptionMenu(main_frame, values=models_available)
        select_model.configure(corner_radius=32,
                               text_color="black", font=CTkFont("Arial", 12, weight="bold"), anchor="center",
                               dropdown_text_color="black", dropdown_font=CTkFont("Arial", 12, weight="bold"),
                               dropdown_fg_color="#1f6aa5", dropdown_hover_color=self.COLOR_PRESETS.BTN_HOVER_COLOR,
                               command=lambda selected_model: [self._setParameter("model", models_available.index(selected_model)), self._reRender(main_frame)])
        select_model.set(models_available[self.model_to_use])
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


    def _reRender(self, main_frame):
        for widget in main_frame.winfo_children():
            widget.destroy()
        
        self._loadContents(main_frame)
    
    def _splitRenderer(self, main_frame):
        if self.model_to_use == 2:
            base_model_renderer = CTkButton(main_frame)
            base_model_renderer.configure(corner_radius=32,
                                            hover_color=self.COLOR_PRESETS.BTN_HOVER_COLOR,
                                            fg_color="transparent", bg_color="transparent",
                                            border_color="white", border_width=1,
                                            text="Show 3D Model DeepLabV3+", text_color="white",
                                            command=lambda: self._openRenderer(0))
            base_model_renderer.place(relx=0.73, rely=0.22, relwidth=0.4, relheight=0.34, anchor="center")
            
            modified_model_renderer = CTkButton(main_frame)
            modified_model_renderer.configure(corner_radius=32,
                                            hover_color=self.COLOR_PRESETS.BTN_HOVER_COLOR,
                                            fg_color="transparent", bg_color="transparent",
                                            border_color="white", border_width=1,
                                            text="Show 3D Model DeepLabV3+ w/ CA & SA", text_color="white",
                                            command=lambda: self._openRenderer(1))
            modified_model_renderer._text_label.configure(wraplength=150)
            modified_model_renderer.place(relx=0.73, rely=0.58, relwidth=0.4, relheight=0.34, anchor="center")
        else:
            model_renderer_frame = CTkButton(main_frame)
            model_renderer_frame.configure(corner_radius=32,
                                        hover_color=self.COLOR_PRESETS.BTN_HOVER_COLOR,
                                        fg_color="transparent", bg_color="transparent",
                                        border_color="white", border_width=1,
                                        text="Show 3D Model", text_color="white",
                                        command=self._openRenderer)
            model_renderer_frame.place(relx=0.73, rely=0.4, relwidth=0.4, relheight=0.7, anchor="center")
    
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
            self.processed_data.clear()
        elif parameter == "model":
            self.model_to_use = value
        
        print(f"Selected {parameter}: {value}")
    
    def _seeSegMaps(self):
        # model_output_path = r"D:\GitHub\deepl_lab\tool\deeplab\floorplan_pred512.pt"
        # model_output_path1 = r"D:\GitHub\deepl_lab\tool\deeplab\floorplan_pred512.pt"
        
        # print(model_outputs.shape)
        # ic(torch.load(model_output_path))
        self.root.seeSegMaps(self.output, self.model_to_use)
        # self.root.seeDetails(self.output, self.labels)
        
    def _seeDetails(self):
        metric_outputs = self.labels
        print(metric_outputs)
        self.root.seeDetails(metric_outputs, self.model_to_use)

    def _generateModel(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img, labels = load_img_and_labels(self.dataset, self.selected_floorplan)
        
        models = []
        
        metrics = []
        outputs = []
        ground_truths = []

        if self.model_to_use < 2:
            model = load_model(
                f"C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/tool/deeplab/deeplab_efficientnet_b2_{'base' if self.model_to_use == 0 else 'ca_sa'}.pt", 
                use_attention=False if self.model_to_use == 0 else True, device=device)
            combined_output, combined_labels, metrics_output = evaluate(model, img, labels)
            metrics.append(metrics_output)
            outputs.append(combined_output)
            ground_truths.append(combined_labels)
            print("Model generated")

        else:
            for i in range(2):
                model = load_model(
                    f"C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/tool/deeplab/deeplab_efficientnet_b2_{'base' if i == 0 else 'ca_sa'}.pt", 
                    use_attention=False if i == 0 else True, device=device)
                models.append(model)
                combined_output, combined_labels, metrics_output = evaluate(model, img, labels)
                metrics.append(metrics_output)
                outputs.append(combined_output)
                if i == 0:
                    ground_truths.append(combined_labels)
                print(f"Model loaded: deeplabv3plus_{model.backbone_name}_{model.attention}")
                print("Model generated")

        
        self.labels = metrics
        self.output = outputs
        self.output.extend(ground_truths)

        # self.Vectorizer(combined_tensor)
        # sample = torch.load(r"D:\GitHub\deepl_lab\tool\deeplab\floorplan_pred512.pt")
        if self.model_to_use < 2:
            renderer = self.RECONSTRUCTION.Renderer()
            vectorizer = self.RECONSTRUCTION.Vectorizer()
            scaled_rooms, scaled_outer_contour, scaled_walls, scaled_icons, icon_quadrilaterals, room_classes = vectorizer.process_data(self.output[0])
            renderer.generate_model(scaled_rooms, scaled_outer_contour, scaled_walls, scaled_icons, icon_quadrilaterals, room_classes)
            self.processed_data.append(renderer)
        else:
            for i in range(2):
                renderer = self.RECONSTRUCTION.Renderer()
                vectorizer = self.RECONSTRUCTION.Vectorizer()
                scaled_rooms, scaled_outer_contour, scaled_walls, scaled_icons, icon_quadrilaterals, room_classes = vectorizer.process_data(self.output[i])
                renderer.generate_model(scaled_rooms, scaled_outer_contour, scaled_walls, scaled_icons, icon_quadrilaterals, room_classes)
                self.processed_data.append(renderer)

    def _openRenderer(self, model_to_use=None):
        # threading.Thread(target=self.Renderer.show_model).start()
        try:
            if self.model_to_use == 2:
                self.processed_data[model_to_use].show_model()
            else:
                self.processed_data[0].show_model()
        except Exception as e:
            if str(e) == "'Renderer' object has no attribute 'window'":
                print("Model not generated yet")