import numpy as np

class Vect:
    
    _instance = None

    I_Xaxis = 0
    I_Yaxis = 1
    I_Zaxis = 2

    def __init__(self):
        pass

    def VECTR_cross(self, a, b, c, d, rdtype):
        nv = np.empty(3, dtype=rdtype)
        nv[0] = (b[1] - a[1]) * (d[2] - c[2]) - (b[2] - a[2]) * (d[1] - c[1])
        nv[1] = (b[2] - a[2]) * (d[0] - c[0]) - (b[0] - a[0]) * (d[2] - c[2])
        nv[2] = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
        return nv

#    def VECTR_abs(self, a, rdtype):
#        l=rdtype(np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]))
#        return l

#    def VECTR_abs(self, a, rdtype):
#        l=np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
#        return l

    def VECTR_abs(self, a, rdtype):
        return np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

#    def VECTR_abs(self, a, rdtype):
#        return np.linalg.norm(a)

    
#    def VECTR_dot(self, a, b, c, d, rdtype):
#        l = rdtype((b[0] - a[0]) * (d[0] - c[0]) +  (b[1] - a[1]) * (d[1] - c[1]) + (b[2] - a[2]) * (d[2] - c[2]))
#        return l

    def VECTR_dot(self, a, b, c, d, rdtype):
        return (b[0] - a[0]) * (d[0] - c[0]) +  (b[1] - a[1]) * (d[1] - c[1]) + (b[2] - a[2]) * (d[2] - c[2])
    
    def VECTR_angle(self, a, b, c, rdtype):
        nvlenC = self.VECTR_dot(b, a, b, c, rdtype)
        nv   = self.VECTR_cross(b, a, b, c, rdtype)    
        nvlenS = self.VECTR_abs(nv, rdtype)
        angle  = np.arctan2(nvlenS, nvlenC)
        return angle
    
    def VECTR_xyz2latlon(self, x, y, z, cnst):
    
        length = np.sqrt(x*x + y*y + z*z)

        # If the vector length is too small, return (lat, lon) = (0, 0)
        if length < cnst.CONST_EPS:
            return 0.0, 0.0

        # Handle special cases where the vector is parallel to the Z-axis
        if z / length >= 1.0:
            return np.arcsin(1.0), 0.0
        elif z / length <= -1.0:
            return np.arcsin(-1.0), 0.0
        else:
            lat = np.arcsin(z / length)

        # Compute horizontal length
        length_h = np.sqrt(x*x + y*y)

        # If horizontal length is too small, set longitude to zero
        if length_h < cnst.CONST_EPS:
            return lat, 0.0

        # Compute longitude using arccos
        if x / length_h >= 1.0:
            lon = np.arccos(1.0)
        elif x / length_h <= -1.0:
            lon = np.arccos(-1.0)
        else:
            lon = np.arccos(x / length_h)

        # Adjust sign based on y value
        if y < 0.0:
            lon = -lon

        return lat, lon

    def VECTR_xyz2latlon_vec(self, x, y, z, cnst): # <added by a.kamiijo on 2025.04.01>
        eps = cnst.CONST_EPS
        length = np.sqrt(x * x + y * y + z * z)
        near_zero = length < eps
        safe_length = np.where(length == 0, 1, length)
        lat = np.arcsin(np.clip(z / safe_length, -1.0, 1.0))
        length_h = np.sqrt(x * x + y * y)
        safe_length_h = np.where(length_h == 0, 1, length_h)
        lon = np.arccos(np.clip(x / safe_length_h, -1.0, 1.0))
        lon = np.where(y < 0.0, -lon, lon)
        lon = np.where(length_h < eps, 0.0, lon)
        lat = np.where(near_zero, 0.0, lat)
        lon = np.where(near_zero, 0.0, lon)
        return lat, lon

    def VECTR_triangle(self, a, b, c, polygon_type, radius, cnst, rdtype):

        #import math

        """
        Compute the area of a triangle on either a plane or a sphere.

        Parameters:
            a (numpy.ndarray): 3D coordinates of vertex A.
            b (numpy.ndarray): 3D coordinates of vertex B.
            c (numpy.ndarray): 3D coordinates of vertex C.
            polygon_type (str): "ON_PLANE" for planar triangles, "ON_SPHERE" for spherical triangles.
            radius (float): Radius of the sphere (if applicable).

        Returns:
            float: The computed area of the triangle.
        """

        # Constants
        #PI = cnst.CONST_PI
        #EPS = 1e-10  # Small epsilon value to prevent division by zero

        # Initialize area
        area = 0.0

        if polygon_type == "ON_PLANE":
            # Compute cross product of vectors AB and AC
            abc = self.VECTR_cross(a, b, a, c, rdtype)
            prd = self.VECTR_abs(abc)  # Magnitude of the cross product
            r = self.VECTR_abs(a)  # Distance from origin

            prd = 0.5 * prd  # Triangle area

            if r < cnst.CONST_EPS:
                print("Zero length?", a)
            else:
                r = 1.0 / r  # Inverse length scaling

            area = prd * r * r * radius * radius

        elif polygon_type == "ON_SPHERE":
            # Compute angles between vectors using dot product (Haversine-like approach)
            o = np.array([0.0, 0.0, 0.0])  # Origin

            len_1 = self.VECTR_angle(a, o, b, rdtype)
            len_2 = self.VECTR_angle(b, o, c, rdtype)
            len_3 = self.VECTR_angle(c, o, a, rdtype)

            # Compute area using l'Huillier's theorem
            len_1 *= 0.5
            len_2 *= 0.5
            len_3 *= 0.5
            s = 0.5 * (len_1 + len_2 + len_3)

            x = np.tan(s) * np.tan(s - len_1) * np.tan(s - len_2) * np.tan(s - len_3)
            x = max(x, 0.0)  # Ensure non-negative values

            area = 4.0 * np.atan(np.sqrt(x)) * radius * radius

        return area

vect = Vect()
#print('instantiated vect')