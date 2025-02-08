class Adm:
    # Basic definition & information

    # Local region and process
    I_l = 1
    I_prc = 2

    # Region ID and direction
    I_RGNID = 1
    I_DIR = 2

    # Identifiers of directions of region edges
    I_SW = 1
    I_NW = 2
    I_NE = 3
    I_SE = 4

    # Identifiers of directions of region vertices
    I_W = 1
    I_N = 2
    I_E = 3
    I_S = 4

    # Identifier of poles (north pole or south pole)
    I_NPL = 1
    I_SPL = 2

    # Identifier of triangle element (i-axis-side or j-axis side)
    ADM_TI = 1
    ADM_TJ = 2

    # Identifier of arc element (i-axis-side, ij-axis side, or j-axis side)
    ADM_AI = 1
    ADM_AIJ = 2
    ADM_AJ = 3

    # Identifier of 1 variable
    ADM_KNONE = 1

    # Dimension of the spatial vector
    ADM_nxyz = 3

    # You can add methods here as needed.
    def ADM_setup(self):
        # This method appears to be a placeholder for now.
        # Add the necessary setup code here.
        pass
