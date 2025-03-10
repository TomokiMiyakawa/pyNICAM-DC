class Grd:
    
    _instance = None
    
    # character length

    #++ Public parameters & variables

    #Indentifiers for the directions in the Cartesian coordinate
    GRD_XDIR = 0
    GRD_YDIR = 1
    GRD_ZDIR = 2

    #Indentifiers for the directions in the spherical coordinate
    I_LAT = 0
    I_LON = 1

#====== Horizontal Grid ======
#
# Grid points ( X: CELL CENTER )
#           .___.
#          /     \
#         .   p   .
#          \ ___ /
#           '   '
#
# Grid points ( Xt: CELL VERTEX )
#           p___p
#          /     \
#         p       p
#          \ ___ /
#           p   p
#
# Grid points ( Xr: CELL ARC )
#           ._p_.
#          p     p
#         .       .
#          p _ _ p
#           ' p '

    # data type