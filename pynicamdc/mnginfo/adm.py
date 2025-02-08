class Adm:
    # Basic definition & information   This adm is zero-based, unlike the original NICAM code

    # Local region and process
    I_l = 0
    I_prc = 1

    # Region ID and direction
    I_RGNID = 0
    I_DIR = 1

    # Identifiers of directions of region edges
    I_SW = 0
    I_NW = 1
    I_NE = 2
    I_SE = 3

    # Identifiers of directions of region vertices
    I_W = 0
    I_N = 1
    I_E = 2
    I_S = 3

    # Identifier of poles (north pole or south pole)
    I_NPL = 0
    I_SPL = 1

    # Identifier of triangle element (i-axis-side or j-axis side)
    ADM_TI = 0
    ADM_TJ = 1

    # Identifier of arc element (i-axis-side, ij-axis side, or j-axis side)
    ADM_AI = 0
    ADM_AIJ = 1
    ADM_AJ = 2

    # Identifier of 1 variable
    ADM_KNONE = 1

    # Dimension of the spatial vector
    ADM_nxyz = 3

    # You can add methods here as needed.
    def ADM_setup(self):
        # This method appears to be a placeholder for now.
        # Add the necessary setup code here.
        pass
