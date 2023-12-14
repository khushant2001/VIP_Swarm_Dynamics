"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - BGM
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import Euler2Rotation


class DrawMav:
    def __init__(self, state, window):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.mav_points, self.mav_meshColors = self.get_points()

        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        self.mav_body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.mav_meshColors,  # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
        window.addItem(self.mav_body)  # add body to plot

    def update(self, state):
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        # draw MAV by resetting mesh using rotated and translated points
        self.mav_body.setMeshData(vertexes=mesh, vertexColors=self.mav_meshColors)

    def rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1, points.shape[1]]))
        return translated_points

    def get_points(self):
        """"
            Points that define the mav, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure C.3
        """
        # points are in NED coordinates
        # define the points on the aircraft following diagram Fig 2.14
        points = 1/10 * np.array([[-1, 1, -1/2],
                           [1, 1, -1/2],
                           [-1, -1, -1/2],
                           [1, -1, -1/2],
                           [-2, 2, -1/2],
                           [2, 2, -1/2],
                           [-2, -2, -1/2],
                           [2, -2, -1/2],
                           [-1, 1, 1/2],
                           [1, 1, 1/2],
                           [-1, -1, 1/2],
                           [1, -1, 1/2],
                           [-2, 2, 1/2],
                           [2, 2, 1/2],
                           [-2, -2, 1/2],
                           [2, -2, 1/2],
                           ]).T

        # scale points for better rendering
        scale = 75
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((20, 3, 4), dtype=np.float32)
        meshColors[0] = yellow
        meshColors[1] = yellow
        meshColors[2] = yellow
        meshColors[3] = yellow
        meshColors[4] = yellow
        meshColors[5] = yellow
        meshColors[6] = yellow
        meshColors[7] = yellow
        meshColors[8] = red
        meshColors[9] = red
        meshColors[10] = red
        meshColors[11] = red
        meshColors[12] = red
        meshColors[13] = red
        meshColors[14] = red
        meshColors[15] = red
        meshColors[16] = green
        meshColors[17] = green
        meshColors[18] = green
        meshColors[19] = green

        return points, meshColors

    def points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points = points.T
        mesh = np.array([
            [points[0], points[4], points[12]],
            [points[0], points[8], points[12]],
            [points[1], points[5], points[13]],
            [points[1], points[9], points[13]],
            [points[2], points[6], points[14]],
            [points[2], points[10], points[14]],
            [points[3], points[7], points[15]],
            [points[3], points[11], points[15]],
            [points[0], points[1], points[9]],
            [points[0], points[8], points[9]],
            [points[1], points[3], points[11]],
            [points[1], points[9], points[11]],
            [points[3], points[2], points[10]],
            [points[3], points[10], points[11]],
            [points[2], points[0], points[8]],
            [points[2], points[8], points[10]],
            [points[0], points[1], points[2]],
            [points[1], points[2], points[3]],
            [points[8], points[9], points[10]],
            [points[9], points[10], points[11]]])

        return mesh
