import numpy as np

class Precision:
    def __init__(self,single):
        self.SP = np.float32(0.0)
        self.DP = np.float64(0.0)
        self.SP_PREC = self.precision(self.SP)
        self.DP_PREC = self.precision(self.DP)

        if single:
            self.RP = self.SP
            self.RP_PREC = self.SP_PREC
        else:
            self.RP = self.DP
            self.RP_PREC = self.DP_PREC

    def precision(self, value):
        return np.finfo(value).bits

# Example usage:
SINGLE =  True  # You can set this to True or False to choose single or double precision
precision_obj = Precision(SINGLE)
print("SP:", precision_obj.SP)
print("DP:", precision_obj.DP)
print("SP_PREC:", precision_obj.SP_PREC)
print("DP_PREC:", precision_obj.DP_PREC)
print("RP:", precision_obj.RP)
print("RP_PREC:", precision_obj.RP_PREC)
