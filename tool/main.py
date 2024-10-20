import presets as PATHS
import reconstruction as RECONSTRUCTION
from ui import *

if __name__ == "__main__":
    rootUI = RootUI()
    rootUI.initializePresets(PATHS, RECONSTRUCTION)
    rootUI.createRoot()
    rootUI.loadMainInterface()
    
    rootUI.mainloop()