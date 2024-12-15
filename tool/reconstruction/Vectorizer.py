import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from presets import OUTPUT
from icecream import ic

# torch.set_printoptions(profile="full")

# sample = torch.load(OUTPUT.PYTORCH_VALUES)
# print(sample.shape)

# # Room segmentation map
# plt.imshow(sample[21])
# plt.axis('off') # Remove the axes
# plt.show()
# # plt.set_title('Room Labels')

# # Icon segmentation map
# plt.imshow(sample[22])
# plt.axis('off')
# # plt.set_title('Icon Labels')

# plt.tight_layout()
# plt.show()

# Heatmaps
# for i in range(21):
#     plt.imshow(sample[i])
#     plt.title(f'Heatmap {i+1}')
#     plt.axis('off')
#     plt.show()

class Vectorizer():
    def __init__(self):
        pass
        # rooms, contours, walls, icons, icon_class, room_class
        
    # Helper function: Extract contours and fit to quadrilateral (4 points)
    def _extract_labeled_room_contours(self, segmentation_map, min_area=0):
        #room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
        unique_labels = np.unique(segmentation_map)
        room_contours = []
        room_classes = []  # Add list to store room classes
        outer_contour = None
        max_area = 12000

        for label in unique_labels:
            room_mask = (segmentation_map == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if area > max_area:
                        outer_contour = approx
                        outer_contour_class = label  # Store class for outer contour
                    else:
                        room_contours.append(approx)
                        room_classes.append(label)  # Store class for each room

        return outer_contour, outer_contour_class, room_contours, room_classes  # Return classes along with contours

    # def get_outer_floorplan_contour(segmentation_map):
    #     outer_contour, _ = extract_labeled_room_contours(segmentation_map)
    #     return outer_contour

    def _line_intersects_rectangle(self, line_start, line_end, rect):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def line_intersects(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        rect_lines = [
            (rect[0], rect[1]),
            (rect[1], rect[2]),
            (rect[2], rect[3]),
            (rect[3], rect[0])
        ]

        for rect_line in rect_lines:
            if line_intersects(line_start, line_end, rect_line[0], rect_line[1]):
                return True

        return False

    def _point_in_rectangle(self, point, rect):
        x, y = point
        return rect[0][0] <= x <= rect[2][0] and rect[0][1] <= y <= rect[2][1]

    def _find_intersection_point(self, line_start, line_end, rect):
        def line_intersection(line1, line2):
            x1, y1 = line1[0]
            x2, y2 = line1[1]
            x3, y3 = line2[0]
            x4, y4 = line2[1]
            
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                return None  # Lines are parallel
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            if 0 <= t <= 1:
                return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            return None

        rect_lines = [
            (rect[0], rect[1]),
            (rect[1], rect[2]),
            (rect[2], rect[3]),
            (rect[3], rect[0])
        ]

        for rect_line in rect_lines:
            intersection = line_intersection((line_start, line_end), rect_line)
            if intersection:
                return intersection
        return None

    def _point_equal(self, p1, p2):
        return np.allclose(p1, p2)

    def _generate_walls(self, room_polygons, icon_quadrilaterals, which, wall_height=10):
        walls = []
        for i, room in enumerate(room_polygons):
            # if which == "outer" and i == 0:
            #     continue
            # elif which == "inner" and i == 10:
            #     continue
            for i in range(len(room)):
                next_i = (i + 1) % len(room)
                wall_start = room[i][0]
                wall_end = room[next_i][0]

                current_point = wall_start
                while not self._point_equal(current_point, wall_end):
                    next_point = wall_end
                    intersecting_icon = None
                    intersection_point = None

                    for icon in icon_quadrilaterals:
                        if self._line_intersects_rectangle(current_point, next_point, icon):
                            intersecting_icon = icon
                            intersection_point = self._find_intersection_point(current_point, next_point, icon)
                            if intersection_point is not None:
                                next_point = intersection_point
                                break

                    if intersecting_icon is not None:
                        # Create wall segment up to the icon
                        wall = [
                            [current_point[0], current_point[1], 0],
                            [next_point[0], next_point[1], 0],
                            [next_point[0], next_point[1], wall_height],
                            [current_point[0], current_point[1], wall_height]
                        ]
                        walls.append(wall)

                        # Find exit point from the icon
                        exit_point = None
                        for icon_point in icon:
                            if not self._point_in_rectangle(icon_point, intersecting_icon) and \
                            self._line_intersects_rectangle(icon_point, wall_end, intersecting_icon):
                                exit_point = icon_point
                                break

                        if exit_point is not None:
                            current_point = exit_point
                        else:
                            # If no suitable exit point found, move to the next corner of the icon
                            icon_index = next((i for i, point in enumerate(icon) if self._point_equal(point, next_point)), None)
                            if icon_index is not None:
                                current_point = icon[(icon_index + 1) % 4]
                            else:
                                # If we can't find the next point in the icon, just move to wall_end to avoid infinite loop
                                current_point = wall_end
                    else:
                        # No intersection, create full wall segment
                        wall = [
                            [current_point[0], current_point[1], 0],
                            [next_point[0], next_point[1], 0],
                            [next_point[0], next_point[1], wall_height],
                            [current_point[0], current_point[1], wall_height]
                        ]
                        walls.append(wall)
                        current_point = next_point

        return walls

    # Helper function: Extract icons (doors, windows, etc.) as quadrilaterals
    def _extract_icon_quadrilaterals(self, icon_map):
        icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]  # Assuming these are your icon classes
        icon_map_8bit = (icon_map * 255).astype(np.uint8)
        _, thresholded = cv2.threshold(icon_map_8bit, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Convert contours to quadrilateral format and include class information
        icon_quads = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            quad = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            class_index = i % len(icon_classes)  # Cycle through class indices
            icon_quads.append((quad, class_index))  # Include class index with the quad
        return icon_quads

    def _scale_room_polygon(self, room_polygon, scale_factor):
        """
        Scale the given room polygon by the scale_factor.
        """
        return [[[point[0][0] * scale_factor, point[0][1] * scale_factor]] for point in room_polygon]

    def _scale_coordinates(self, coordinates, scale_factor):
        """
        Scale the given coordinates by the scale_factor.
        """
        return [[coord[0] * scale_factor, coord[1] * scale_factor] for coord in coordinates]

    def _scale_wall(self, wall, scale_factor):
        """
        Scale the given wall coordinates by the scale_factor.
        """
        return [[coord[0] * scale_factor, coord[1] * scale_factor, coord[2] * scale_factor] for coord in wall]

    def _floorplan_to_vectorized_output(self, torch_file):
        """
        torch_file: Loaded PyTorch tensor with 23 channels [23, 256, 256]
        """
        scale_factor = 3
        
        # Load the PyTorch tensor file
        segmentation_maps = torch_file
        # print(segmentation_maps.shape)

        # Extract room segmentation from Channel 21 (index 20)
        room_segmentation = segmentation_maps[0].cpu().numpy()
        outer_contour, outer_contour_class, room_polygons, room_classes = self._extract_labeled_room_contours(room_segmentation)
        outer_contour = [outer_contour] if outer_contour is not None else []
        # room_polygons = extract_labeled_room_contours(room_segmentation)
        # all_contours = [outer_contour] + room_contours if outer_contour is not None else room_contours
        
        # Extract door, window, and furniture quadrilaterals from Channel 22 (index 21)
        icon_map = segmentation_maps[1].cpu().numpy()
        icon_quadrilaterals = self._extract_icon_quadrilaterals(icon_map)

        # vectorized_rooms = [poly for poly in room_polygons]
        outer_walls = self._generate_walls(outer_contour, [], "outer", 10)
        inner_walls = self._generate_walls(room_polygons, [], "inner", 10)
        
        walls = inner_walls + outer_walls
        # vectorized_icons = [quad for quad in icon_quadrilaterals]
        
        scaled_rooms = [self._scale_room_polygon(poly, scale_factor) for poly in room_polygons]
        scaled_outer_contour = [self._scale_room_polygon(contour, scale_factor) for contour in outer_contour]
        scaled_walls = [self._scale_wall(wall, scale_factor) for wall in walls]
        scaled_icons = [self._scale_coordinates(quad[0], scale_factor) for quad in icon_quadrilaterals]

        return scaled_rooms, scaled_outer_contour, scaled_walls, scaled_icons, icon_quadrilaterals, room_classes

    def process_data(self, data):
        # Your data processing logic
        result = self._floorplan_to_vectorized_output(data)
        return result