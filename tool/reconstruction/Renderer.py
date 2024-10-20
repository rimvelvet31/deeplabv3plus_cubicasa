import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QFont, QPainter  # Import QFont and QPainter
from OpenGL.GL import *
from OpenGL.GLU import *

from icecream import ic

class FloorPlanViewer(QOpenGLWidget):
    def __init__(self, rooms, contours, walls, icons, icon_class, room_class):
        super().__init__()
        self.rooms = rooms
        self.contours = contours
        self.icons = icons
        self.icon_class = icon_class
        self.room_class = room_class
        self.walls = walls
        self.show_rooms = True
        self.show_walls = True
        self.show_icons = True
        self.rot_x = 0
        self.rot_y = 0
        self.last_x = 0
        self.last_y = 0
        self.display_list = None
        self.offset_factor = 1.0
    
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        self.create_display_list()

    def create_display_list(self):
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        # Render the floor (this remains unchanged)
        glColor3f(0.8, 0.8, 0.8)
        glBegin(GL_QUADS)
        glVertex3f(-250*self.offset_factor, -3, -250*self.offset_factor)
        glVertex3f(250*self.offset_factor, -3, -250*self.offset_factor)
        glVertex3f(250*self.offset_factor, -3, 250*self.offset_factor)
        glVertex3f(-250*self.offset_factor, -3, 250*self.offset_factor)
        glEnd()
        
        # Render rooms
        room_colors = [  # Define colors for each room class
                (0, 0, 0),    # Background - Black
                (0.5, 0.5, 0.5),    # Outdoor - Gray
                (0.8, 0.8, 0.8),    # Wall - Light Gray
                (1, 1, 0),    # Kitchen - Yellow
                (0, 1, 0),    # Living Room - Green
                (1, 0, 0),    # Bed Room - Red
                (0, 0, 1),    # Bath - Blue
                (1, 0, 1),    # Entry - Magenta
                (0.5, 0, 0.5),  # Railing - Purple
                (0.7, 0.7, 0),  # Storage - Olive
                (0.2, 0.2, 0.2),  # Garage - Dark Gray
                (0, 1, 1)   # Undefined - Cyan
            ]
        if self.show_rooms:
            for i, room in enumerate(self.rooms):  # Use enumerate to get both index and room
                class_index = self.room_class[i]  # Get class index for the current room
                glColor3f(*room_colors[int(class_index)])  # Set color based on class
                self.draw_polygon(room, "room")
                
            for contour in self.contours:
                glColor3f(0.9, 0.9, 0.9)
                self.draw_polygon(contour, "contour")
        
        if self.show_icons:
            # Define colors for each icon class
            icon_colors = [
                (0, 0, 0),    # No Icon - Black
                (0, 1, 1),    # Window - Cyan
                (0, 150/255, 75/255),    # Door - Brown
                (0, 0, 1),    # Closet - Blue
                (1, 1, 0),    # Electrical Applience - Yellow
                (1, 0, 1),    # Toilet - Magenta
                (1, 1, 1),    # Sink - White
                (0.5, 0.5, 0.5),  # Sauna Bench - Gray
                (1, 0.5, 0),  # Fire Place - Orange
                (0.9, 0.9, 0.9),  # Bathtub - White
                (0, 0.5, 1)   # Chimney - Sky Blue
            ]

            for i, (_, class_index) in enumerate(self.icon_class):  # Unpack the tuple
                glColor3f(*icon_colors[class_index])  # Set color based on class
                self.draw_quadrilateral(self.icons[i])
        
        if self.show_walls:
            glColor3f(0.9, 0.9, 0.9)
            for wall in self.walls:
                self.draw_wall(wall)
        
        glEndList()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, -25, -1000)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)

        glCallList(self.display_list)
        
    def toggle_rooms(self):
        self.show_rooms = not self.show_rooms
        self.create_display_list()
        self.update()

    def toggle_icons(self):
        self.show_icons = not self.show_icons
        self.create_display_list()
        self.update()

    def toggle_walls(self):
        self.show_walls = not self.show_walls
        self.create_display_list()
        self.update()
        
    def draw_wall(self, quad):
        """Draws a quadrilateral given its 4 corner points."""
        glBegin(GL_QUADS)
        for i in range(len(quad)):
            p1 = quad[i]
            p2 = quad[(i + 1) % len(quad)]
            glVertex3f(p1[0] - 250*self.offset_factor, 0, p1[1] - 250*self.offset_factor)
            glVertex3f(p1[0] - 250*self.offset_factor, 50, p1[1] - 250*self.offset_factor)
            glVertex3f(p2[0] - 250*self.offset_factor, 50, p2[1] - 250*self.offset_factor)
            glVertex3f(p2[0] - 250*self.offset_factor, 0, p2[1] - 250*self.offset_factor)
        glEnd()
        
    def draw_quadrilateral(self, quad):
        """Draws a quadrilateral with a roof."""
        
        # Calculate the center point of the quadrilateral
        center_x = sum(point[0] for point in quad) / 4
        center_y = sum(point[1] for point in quad) / 4
        
        # Draw the roof (a sloped top)
        glBegin(GL_QUADS)
        for i in range(len(quad)):
            p1 = quad[i]
            p2 = quad[(i + 1) % len(quad)]
            glVertex3f(p1[0] - 250*self.offset_factor, 50, p1[1] - 250*self.offset_factor)  # Top of wall
            glVertex3f(center_x - 250*self.offset_factor, 50, center_y - 250*self.offset_factor)  # Peak of roof
            glVertex3f(p2[0] - 250*self.offset_factor, 50, p2[1] - 250*self.offset_factor)  # Top of next wall
            glVertex3f(p2[0] - 250*self.offset_factor, 50, p2[1] - 250*self.offset_factor)  # Top of next wall
        glEnd()

        # Draw the walls
        glBegin(GL_QUADS)
        for i in range(len(quad)):
            p1 = quad[i]
            p2 = quad[(i + 1) % len(quad)]
            glVertex3f(p1[0] - 250*self.offset_factor, 0, p1[1] - 250*self.offset_factor)  # Base of wall
            glVertex3f(p1[0] - 250*self.offset_factor, 50, p1[1] - 250*self.offset_factor)  # Top of wall
            glVertex3f(p2[0] - 250*self.offset_factor, 50, p2[1] - 250*self.offset_factor)  # Top of next wall
            glVertex3f(p2[0] - 250*self.offset_factor, 0, p2[1] - 250*self.offset_factor)  # Base of next wall
        glEnd()
        
    def draw_polygon(self, polygon, which):
        """Draws a polygon given its vertices, with improved handling for curves."""
        elevation = 0
        glBegin(GL_TRIANGLE_FAN)
        # Calculate centroid
        centroid_x = sum(vertex[0][0] for vertex in polygon) / len(polygon)
        centroid_y = sum(vertex[0][1] for vertex in polygon) / len(polygon)
        
        if which == "contour":
            elevation = -2
        
        # Start with centroid
        glVertex3f(centroid_x - 250*self.offset_factor, elevation, centroid_y - 250*self.offset_factor)
        
        for vertex in polygon:
            x, y = vertex[0][0], vertex[0][1]
            glVertex3f(x - 250*self.offset_factor, elevation, y - 250*self.offset_factor)
        
        # Close the polygon
        x, y = polygon[0][0][0], polygon[0][0][1]
        glVertex3f(x - 250*self.offset_factor, elevation, y - 250*self.offset_factor)
        
        glEnd()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 2000.0)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.last_x = event.x()
        self.last_y = event.y()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_x
        dy = event.y() - self.last_y

        self.rot_x = clamp(self.rot_x + dy, -90, 90)
        self.rot_y += dx

        self.last_x = event.x()
        self.last_y = event.y()
        self.update()

def clamp(value, minimum, maximum):
    return max(min(value, maximum), minimum)

class MainWindow(QMainWindow):
    def __init__(self, rooms, contours, walls, icons, icon_class, room_class):
        super().__init__()
        self.setWindowTitle("3D Floor Plan Viewer")
        
        # Initialize main widget
        self.viewer = FloorPlanViewer(rooms, contours, walls, icons, icon_class, room_class)
        
        # Initialize toggle buttons
        toggle_room_btn = QPushButton("Toggle Rooms", self)
        toggle_room_btn.clicked.connect(self.viewer.toggle_rooms)
        
        toggle_wall_btn = QPushButton("Toggle Walls", self)
        toggle_wall_btn.clicked.connect(self.viewer.toggle_walls)
        
        toggle_icon_btn = QPushButton("Toggle Icons", self)
        toggle_icon_btn.clicked.connect(self.viewer.toggle_icons)
        
        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.viewer)
        layout.addWidget(toggle_room_btn)
        layout.addWidget(toggle_wall_btn)
        layout.addWidget(toggle_icon_btn)
        
        # Set central widget with layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setGeometry(100, 100, 800, 600)

class Renderer():
    def __init__(self):
        pass

        # self.show_model()
        # self.window.show()
    def generate_model(self, rooms, contours, walls, icons, icon_class, room_class):
        self.rooms = rooms
        self.contours = contours
        self.walls = walls
        self.icons = icons
        self.icon_class = icon_class
        self.room_class = room_class
        
        self.app = QApplication(sys.argv)
        self.window = MainWindow(self.rooms, self.contours, self.walls, self.icons, self.icon_class, self.room_class)

    def show_model(self):
        if self.window is None:
            return
        self.window.show()
