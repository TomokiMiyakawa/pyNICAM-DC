import numpy as np

class Precision:
    
    _instance = None

    def __init__(self,single):
        self.SP = np.float32(1.234567890123456789012345678901)
        self.DP = np.float64(1.234567890123456789012345678901)
        self.SP_PREC = self.precision(self.SP)
        self.DP_PREC = self.precision(self.DP)

        if single:
            self.RP = self.SP
            self.RP_PREC = self.SP_PREC
            self.rdtype = np.float32
        else:
            self.RP = self.DP
            self.RP_PREC = self.DP_PREC
            self.rdtype = np.float64

    def precision(self, value):   
        return np.finfo(value).bits

# Example usage:
#SINGLE =  True  # You can set this to True or False to choose single or double precision
#precision_obj = Precision(SINGLE)
#print("SP:", precision_obj.SP)
#print("DP:", precision_obj.DP)
#print("SP_PREC:", precision_obj.SP_PREC)
#print("DP_PREC:", precision_obj.DP_PREC)
#print("RP:", precision_obj.RP)
#print("RP_PREC:", precision_obj.RP_PREC)
