import numpy as np

class Vect:
    

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
    
    def VECTR_xyz2latlon(self, x, y, z, cnst, rdtype):
    
        length = np.sqrt(x*x + y*y + z*z)

        # If the vector length is too small, return (lat, lon) = (0, 0)
        if length < cnst.CONST_EPS:
            return rdtype(0.0), rdtype(0.0)

        # Handle special cases where the vector is parallel to the Z-axis
        if z / length >= rdtype(1.0):
            return np.arcsin(rdtype(1.0)), rdtype(0.0)
        elif z / length <= -rdtype(1.0):
            return np.arcsin(-rdtype(1.0)), rdtype(0.0)
        else:
            lat = np.arcsin(z / length)

        # Compute horizontal length
        length_h = np.sqrt(x*x + y*y)

        # If horizontal length is too small, set longitude to zero
        if length_h < cnst.CONST_EPS:
            return lat, rdtype(0.0)

        # Compute longitude using arccos
        if x / length_h >= rdtype(1.0):
            lon = np.arccos(rdtype(1.0))
        elif x / length_h <= -rdtype(1.0):
            lon = np.arccos(-rdtype(1.0))
        else:
            lon = np.arccos(x / length_h)

        # Adjust sign based on y value
        if y < rdtype(0.0):
            lon = -lon

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
        area = rdtype(0.0)

        if polygon_type == "ON_PLANE":
            # Compute cross product of vectors AB and AC
            abc = self.VECTR_cross(a, b, a, c, rdtype)
            prd = self.VECTR_abs(abc)  # Magnitude of the cross product
            r = self.VECTR_abs(a)  # Distance from origin

            prd = rdtype(0.5) * prd  # Triangle area

            if r < cnst.CONST_EPS:
                print("Zero length?", a)
            else:
                r = rdtype(1.0) / r  # Inverse length scaling

            area = prd * r * r * radius * radius

        elif polygon_type == "ON_SPHERE":
            # Compute angles between vectors using dot product (Haversine-like approach)
            o = np.array([rdtype(0.0), rdtype(0.0), rdtype(0.0)])  # Origin

            len_1 = self.VECTR_angle(a, o, b, rdtype)
            len_2 = self.VECTR_angle(b, o, c, rdtype)
            len_3 = self.VECTR_angle(c, o, a, rdtype)

            # Compute area using l'Huillier's theorem
            len_1 *= rdtype(0.5)
            len_2 *= rdtype(0.5)
            len_3 *= rdtype(0.5)
            s = rdtype(0.5) * (len_1 + len_2 + len_3)

            x = np.tan(s) * np.tan(s - len_1) * np.tan(s - len_2) * np.tan(s - len_3)
            x = max(x, rdtype(0.0))  # Ensure non-negative values

            area = rdtype(4.0) * np.atan(np.sqrt(x)) * radius * radius

        return area

    # ------------------------------------------------------------------
    # Array (component-first: axis 0 = x,y,z; trailing axes = grid) versions
    # of the helpers above, for whole-(i,j)-slice setup vectorization. Each
    # output element is the SAME scalar expression of the SAME inputs as the
    # scalar form, so results are BIT-IDENTICAL to the per-(i,j) loop.
    # ------------------------------------------------------------------
    def _cross_arr(self, a, b, c, d):
        # array form of VECTR_cross: (b-a) x (d-c) -- same 3 expressions
        n0 = (b[1] - a[1]) * (d[2] - c[2]) - (b[2] - a[2]) * (d[1] - c[1])
        n1 = (b[2] - a[2]) * (d[0] - c[0]) - (b[0] - a[0]) * (d[2] - c[2])
        n2 = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
        return np.stack((n0, n1, n2), axis=0)

    def VECTR_angle_vec(self, a, b, c, rdtype):
        # array form of VECTR_angle (reuses VECTR_dot/VECTR_abs verbatim)
        nvlenC = self.VECTR_dot(b, a, b, c, rdtype)
        nv = self._cross_arr(b, a, b, c)
        nvlenS = self.VECTR_abs(nv, rdtype)
        return np.arctan2(nvlenS, nvlenC)

    def VECTR_triangle_vec(self, a, b, c, polygon_type, radius, cnst, rdtype):
        # array form of VECTR_triangle, ON_SPHERE only (the path used by GMTR_*).
        o = np.zeros(3, dtype=rdtype)  # Origin (broadcasts as 0 per component)
        len_1 = self.VECTR_angle_vec(a, o, b, rdtype)
        len_2 = self.VECTR_angle_vec(b, o, c, rdtype)
        len_3 = self.VECTR_angle_vec(c, o, a, rdtype)
        len_1 = len_1 * rdtype(0.5)
        len_2 = len_2 * rdtype(0.5)
        len_3 = len_3 * rdtype(0.5)
        s = rdtype(0.5) * (len_1 + len_2 + len_3)
        x = np.tan(s) * np.tan(s - len_1) * np.tan(s - len_2) * np.tan(s - len_3)
        x = np.maximum(x, rdtype(0.0))  # Ensure non-negative values
        area = rdtype(4.0) * np.arctan(np.sqrt(x)) * radius * radius
        return area

    def VECTR_xyz2latlon_vec(self, x, y, z, cnst, rdtype):
        # array form of VECTR_xyz2latlon: branch-by-branch via np.where, each
        # branch the same float expression as the scalar -> bit-identical.
        one = rdtype(1.0); zero = rdtype(0.0)
        length = np.sqrt(x * x + y * y + z * z)
        small = length < cnst.CONST_EPS
        safe_len = np.where(small, one, length)             # avoid 0-division (masked out)
        zr = z / safe_len
        mid = (zr < one) & (zr > -one)
        lat = np.where(zr >= one, np.arcsin(one),
              np.where(zr <= -one, np.arcsin(-one),
                       np.arcsin(np.where(mid, zr, zero))))   # inner arcsin only where valid
        length_h = np.sqrt(x * x + y * y)
        small_h = length_h < cnst.CONST_EPS
        safe_lh = np.where(small_h, one, length_h)
        xr = x / safe_lh
        midx = (xr < one) & (xr > -one)
        lon = np.where(xr >= one, np.arccos(one),
              np.where(xr <= -one, np.arccos(-one),
                       np.arccos(np.where(midx, xr, zero))))
        lon = np.where(y < zero, -lon, lon)                  # sign by y (only reached if length_h>=EPS)
        lon = np.where(small_h, zero, lon)                   # length_h<EPS -> lon=0
        lat = np.where(small, zero, lat)                     # length<EPS -> (0,0)
        lon = np.where(small, zero, lon)
        return lat, lon

vect = Vect()
#print('instantiated vect')