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


vect = Vect()
print('instantiated vect')