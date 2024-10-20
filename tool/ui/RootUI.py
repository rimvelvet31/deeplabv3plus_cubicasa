from customtkinter import *
from ui.MainInterface import MainInterface
from ui.MetricsUI import MetricsUI

class RootUI(CTk):
    def __init__(self):
        CTk.__init__(self)
        
        self.PATHS = None
        self.RECONSTRUCTION = None
    
    def initializePresets(self, PATHS, RECONSTRUCTION):
        self.PATHS = PATHS
        self.RECONSTRUCTION = RECONSTRUCTION

    def createRoot(self):
        self.minsize(400, 650)
        self.maxsize(400, 650)
        self.config(bg=self.PATHS.COLOR_PRESETS.BG_COLOR)
        self.winfo_toplevel().title("3D Model Generation")
    
    def loadMainInterface(self):
        self.main_interface = MainInterface(self, self.PATHS, self.RECONSTRUCTION)

    def seeDetails(self, tensor, output):
        MetricsUI(CTkToplevel(self), tensor, output)
        