import os
import torch

from customtkinter import *
from tkinter import filedialog
from PIL import Image

from icecream import ic

from loss_fn import load_model, load_dataset, load_img_and_labels, evaluate


class MainInterface():
    def __init__(self, root, PATHS, RECONSTUCTION):
        self.COLOR_PRESETS = PATHS.COLOR_PRESETS
        self.IMAGE_PATHS = PATHS.IMAGE_PATHS
        
        self.Renderer = RECONSTUCTION.Renderer()
        self.Vectorizer = RECONSTUCTION.Vectorizer()
        
        self.selected_floorplan = None
        self.model_to_use = None
        self.run_both_models = False

        self.output = None
        self.labels = None

        self.root = root

        self.dataset = load_dataset('C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/data/cubicasa5k/')
        
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
        upload_photo.place(relx=0.5, rely=0.18, relwidth=0.9, relheight=0.3, anchor="center")
        
        generate_model_btn = CTkButton(main_frame, text="Generate 3D Model", text_color="black", font=CTkFont("Arial", 12, weight="bold"))
        generate_model_btn.configure(width=350,
                                     corner_radius=32,
                                     image=self._createCtkImage(self.IMAGE_PATHS.GENERATE_MODEL_ICON, size=(12, 16)),
                                     command=self._generateModel)
        generate_model_btn.place(relx=0.5, rely=0.36, anchor="center")
        
        model_renderer_frame = CTkButton(main_frame)
        model_renderer_frame.configure(corner_radius=32,
                                       hover_color=self.COLOR_PRESETS.BTN_HOVER_COLOR,
                                       fg_color="transparent", bg_color="transparent",
                                       border_color="white", border_width=1,
                                       text="Show 3D Model", text_color="white",
                                       command=self._openRenderer)
        model_renderer_frame.place(relx=0.5, rely=0.595, relwidth=0.9, relheight=0.4, anchor="center")
        
        models_available = ["DeepLabv3+", "DeepLabv3+ (w/ CA & SA)"]
        select_model = CTkOptionMenu(main_frame, values=models_available)
        select_model.configure(corner_radius=32,
                               width=240,
                               text_color="black", font=CTkFont("Arial", 12, weight="bold"), anchor="center",
                               dropdown_text_color="black", dropdown_font=CTkFont("Arial", 12, weight="bold"),
                               dropdown_fg_color="#1f6aa5", dropdown_hover_color=self.COLOR_PRESETS.BTN_HOVER_COLOR,
                               command=lambda selected_model: self._setParameter("model", models_available.index(selected_model)))
        select_model.place(relx=0.5, rely=0.83, anchor="center")
        
        # export_btn = CTkButton(main_frame, 
        #                text="Export", 
        #                text_color="black", 
        #                font=CTkFont("Arial", 12, weight="bold"),
        #                image=self._createCtkImage(self.IMAGE_PATHS.EXPORT_ICON, size=(12, 12)),
        #                width=120, height=30,
        #                corner_radius=32)
        # export_btn.place(relx=0.35, rely=0.88, anchor="center")

        see_details_btn = CTkButton(main_frame, 
                                    text="See Details", 
                                    text_color="black", 
                                    font=CTkFont("Arial", 12, weight="bold"),
                                    image=self._createCtkImage(self.IMAGE_PATHS.SEE_DETAILS_ICON, size=(12, 12)),
                                    width=120, height=30,
                                    corner_radius=32,
                                    command=self._seeDetails)
        see_details_btn.place(relx=0.5, rely=0.88, anchor="center")
        
        use_both_models = CTkCheckBox(main_frame,
                                      text="Run both models at the same time?")
        use_both_models.configure(command=lambda: [select_model.configure(state="disabled" if use_both_models.get() else "normal"), self._setParameter("useBothModels", use_both_models.get())])
        use_both_models.place(relx=0.5, rely=0.95, anchor="center")

        # compare_btn = CTkButton(main_frame, 
        #                         text="Compare", 
        #                         text_color="black", 
        #                         font=CTkFont("Arial", 12, weight="bold"),
        #                         image=self._createCtkImage(self.IMAGE_PATHS.COMPARE_IMAGE_ICON, size=(12, 12)),
        #                         width=120, height=30,
        #                         corner_radius=32)
        # compare_btn.place(relx=0.81, rely=0.95, anchor="center")
    
    def _createCtkImage(self, image_path, size):
        return CTkImage(Image.open(image_path), size=size)
    
    def _uploadPhoto(self, button_widget):
        # selected_floorplan = filedialog.askopenfilename(title="Select a floorplan image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])

        def pop_file(filepath):
            path, _ = os.path.split(filepath)
            return path
        
        selected_floorplan = filedialog.askopenfilename(title="Select a floorplan image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        
        path = pop_file(selected_floorplan) + "/"
        
        if selected_floorplan == "":
            return
        
        uploaded_image = Image.open(selected_floorplan)
        
        button_height = button_widget.winfo_height()
        width, height = uploaded_image.size
        new_width = int((width * button_widget.winfo_height()) / height)
        
        ctk_image = CTkImage(light_image=uploaded_image, dark_image=uploaded_image, size=(new_width, button_height))
        
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
    
    def _seeDetails(self):
        self.root.seeDetails(self.output, self.labels)

    def _generateModel(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = load_model(
            f'C:/Users/Red/Documents/GitHub/deeplabv3plus_cubicasa/tool/deeplab/best_checkpoint_mobilenetv2_{'base' if self.model_to_use == 0 else 'ca_sa'}.pt', 
            use_attention=False if self.model_to_use == 0 else True, device=device)
        
        print(f"Model loaded: deeplabv3plus_{model.backbone_name}_{model.attention}")

        img, labels = load_img_and_labels(self.dataset, self.selected_floorplan)
        # print(img.shape, labels.shape)

        combined_tensor, metrics_output = evaluate(model, img, labels)
        
        self.labels = combined_tensor
        self.output = metrics_output

        # self.Vectorizer(combined_tensor)
        # sample = torch.load(r"C:\Users\Red\Documents\GitHub\deeplabv3plus_cubicasa\tool\deeplab\floorplan_pred256.pt")
        scaled_rooms, scaled_outer_contour, scaled_walls, scaled_icons, icon_quadrilaterals, room_classes = self.Vectorizer.process_data(combined_tensor)
        self.Renderer.generate_model(scaled_rooms, scaled_outer_contour, scaled_walls, scaled_icons, icon_quadrilaterals, room_classes)
        print("Model generated")

    def _openRenderer(self):
        self.Renderer.show_model()