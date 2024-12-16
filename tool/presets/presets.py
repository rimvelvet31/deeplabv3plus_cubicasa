from pathlib import Path

BASE_PATH = Path.cwd() / "tool" / "assets"

class COLOR_PRESETS():
    BG_COLOR = "#090d12"
    BTN_HOVER_COLOR = "#141c26"

class IMAGE_PATHS():
    CAMERA_ICON = BASE_PATH / "camera_icon.png"
    GENERATE_MODEL_ICON = BASE_PATH / "generate_model_icon.png"
    EXPORT_ICON = BASE_PATH / "export_icon.png"
    SEE_DETAILS_ICON = BASE_PATH / "see_details_icon.png"
    COMPARE_IMAGE_ICON = BASE_PATH / "compare_image_icon.png"
    MAP_ICON = BASE_PATH / "map_icon.png"
    
class OUTPUT():
    PYTORCH_VALUES = "D:/JECKO/anything besides games/programming projs/pythonproj/THESIS/deeplab/floorplan_pred256.pt"