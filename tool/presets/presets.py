import os

BASE_PATH = r"C:\Users\Red\Documents\GitHub\deeplabv3plus_cubicasa\tool\assets"

class COLOR_PRESETS():
    BG_COLOR = "#090d12"
    BTN_HOVER_COLOR = "#141c26"

class IMAGE_PATHS():
    CAMERA_ICON = os.path.join(BASE_PATH, "camera_icon.png")
    GENERATE_MODEL_ICON = os.path.join(BASE_PATH, "generate_model_icon.png")
    EXPORT_ICON = os.path.join(BASE_PATH, "export_icon.png")
    SEE_DETAILS_ICON = os.path.join(BASE_PATH, "see_details_icon.png")
    COMPARE_IMAGE_ICON = os.path.join(BASE_PATH, "compare_image_icon.png")
    
class OUTPUT():
    PYTORCH_VALUES = "D:/JECKO/anything besides games/programming projs/pythonproj/THESIS/deeplab/floorplan_pred256.pt"