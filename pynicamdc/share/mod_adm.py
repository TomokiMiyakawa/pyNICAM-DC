class ADM:
    # Basic definition & information
    i_l = 1   # local region
    i_prc = 2   # process

    i_rgnid = 1 # region id
    i_dir = 2   # direction

    # Identifiers of directions of region edges
    i_sw = 1    # south west
    i_nw = 2    # north west
    i_ne = 3    # north east
    i_se = 4    # south east

    # Identifiers of directions of region vertices
    i_w = 1     # west
    i_n = 2     # north
    i_e = 3     # east
    i_s = 4     # south

    # Identifier of poles (north pole or south pole)
    i_npl = 1   # north pole
    i_spl = 2   # south pole

    # Identifier of triangle element (i-axis-side or j-axis side)
    adm_ti = 1
    adm_tj = 2

    # Identifier of arc element (i-axis-side, ij-axis side, or j-axis side)
    adm_ai  = 1
    adm_aij = 2
    adm_aj  = 3

    # Identifier of 1 variable
    adm_knone = 1

    # Dimension of the spatial vector
    adm_nxyz = 3

    @staticmethod
    def adm_setup():
        # This is a placeholder for the ADM_setup function.
        # Fill this in with the appropriate logic.
        pass

