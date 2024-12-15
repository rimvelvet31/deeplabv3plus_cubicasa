from customtkinter import *
from ui.MainInterface import MainInterface
from ui.MetricsUI import SegmentationUI, MetricsUI

class RootUI(CTk):
    def __init__(self):
        CTk.__init__(self)
        
        self.PATHS = None
        self.RECONSTRUCTION = None
    
    def initializePresets(self, PATHS, RECONSTRUCTION):
        self.PATHS = PATHS
        self.RECONSTRUCTION = RECONSTRUCTION

    def createRoot(self):
        self.minsize(600, 550)
        self.maxsize(600, 550)
        self.config(bg=self.PATHS.COLOR_PRESETS.BG_COLOR)
        self.winfo_toplevel().title("3D Model Generation")
    
    def loadMainInterface(self):
        self.main_interface = MainInterface(self, self.PATHS, self.RECONSTRUCTION)

    def seeSegMaps(self, output, model):
        SegmentationUI(CTkToplevel(self), output, model, self.PATHS.COLOR_PRESETS)

    def seeDetails(self, tensor, model):
        MetricsUI(CTkToplevel(self), tensor, model, self.PATHS.COLOR_PRESETS)