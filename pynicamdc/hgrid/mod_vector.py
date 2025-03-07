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

    def VECTR_abs(self, a, rdtype):
        l=rdtype(np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]))
        return l

    
    def VECTR_dot(self, a, b, c, d, rdtype):
        l = rdtype((b[0] - a[0]) * (d[0] - c[0]) +  (b[1] - a[1]) * (d[1] - c[1]) + (b[2] - a[2]) * (d[2] - c[2]))
        return l
    
    def VECTR_angle(self, a, b, c, rdtype):
        nvlenC = self.VECTR_dot(b, a, b, c, rdtype)
        nv   = self.VECTR_cross(b, a, b, c, rdtype)    
        nvlenS = self.VECTR_abs(nv, rdtype)
        angle  = np.arctan2(nvlenS, nvlenC)
        return angle
    
vect = Vect()
print('instantiated vect')