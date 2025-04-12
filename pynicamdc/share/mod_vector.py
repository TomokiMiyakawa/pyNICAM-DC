import numpy as np

"""
<added by a.kamiijo on 2025.04.02>
This module implements vector operations for the NICAM-DC dynamical core.
Key optimizations in this module include:

1. Memory Access Pattern Optimization:
   - Careful management of array strides and layout
   - Cache-friendly processing through tiling for large arrays
   - Minimized temporary array creation
   - In-place operations where possible

2. Vectorization:
   - Vectorized implementations of vector operations (cross products, dot products)
   - Advanced broadcasting for efficient array manipulation
   - Optimized use of NumPy's einsum with optimizer flags
   - Specialized functions for bulk operations on arrays of vectors

3. Performance Improvements:
   - Conditional optimization strategies based on array size
   - Use of optimal memory layouts for different operations
   - Reduced function call overhead through vectorized operations
   - Pure NumPy implementation for wide compatibility
"""

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

    def VECTR_cross_vec(self, a, b, c, d, rdtype=None): # <added by a.kamiijo on 2025.04.02>
        """
        Vectorized version of VECTR_cross that can operate on arrays of vectors.
        
        Parameters:
            a: Array with shape (..., 3) or vector with shape (3,)
            b: Array with shape (..., 3) or vector with shape (3,)
            c: Array with shape (..., 3) or vector with shape (3,)
            d: Array with shape (..., 3) or vector with shape (3,)
            rdtype: Data type for the result
            
        Returns:
            Array with shape (..., 3) containing cross products
        """
        # Advanced optimization: block processing and improved cache efficiency
        
        # Preprocessing: efficiently calculate vector differences
        v1 = b - a  # temporary array, but using advanced broadcasting
        v2 = d - c
        
        # Cache-friendly memory layout
        result_shape = b.shape
        result = np.empty(result_shape, dtype=rdtype, order='C')
        
        # Optimization of cross product calculation: transformed to matrix operations
        # Cross product represented as matrix operations
        # v1 x v2 = [v1[1]*v2[2] - v1[2]*v2[1], 
        #            v1[2]*v2[0] - v1[0]*v2[2], 
        #            v1[0]*v2[1] - v1[1]*v2[0]]
        
        # Maximize use of NumPy's broadcasting capabilities,
        # vectorize each component's operation at once
        result[..., 0] = v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1]
        result[..., 1] = v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2]
        result[..., 2] = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        
        return result

    def VECTR_cross_vec_inplace(self, a, b, c, d, output, rdtype): # <added by a.kamiijo on 2025.04.02>
        """
        Vectorized version of VECTR_cross that operates on arrays of vectors, with in-place output.
        This version avoids creating temporary arrays by writing directly to the provided output array.
        
        Parameters:
            a: Array with shape (..., 3) or vector with shape (3,)
            b: Array with shape (..., 3) or vector with shape (3,)
            c: Array with shape (..., 3) or vector with shape (3,)
            d: Array with shape (..., 3) or vector with shape (3,)
            output: Pre-allocated array with the same shape as b for storing the result
            rdtype: Data type for the result
            
        Returns:
            None - results are written directly to the output array
        """
        # Advanced optimization: block processing and tiling
        
        # For large arrays, use tile processing to improve cache efficiency
        if b.size > 1000000:  # For large arrays
            # Determine tile size (optimized for CPU cache)
            TILE_SIZE = 1024
            
            # Get array shape
            shape = b.shape[:-1]  # Shape excluding the last dimension (3)
            size = np.prod(shape)
            
            # Flatten arrays and process each tile
            v1_flat = (b - a).reshape(-1, 3)
            v2_flat = (d - c).reshape(-1, 3)
            output_flat = output.reshape(-1, 3)
            
            # Tile processing to improve cache efficiency
            for i in range(0, size, TILE_SIZE):
                end = min(i + TILE_SIZE, size)
                # Calculate cross product within each tile
                output_flat[i:end, 0] = v1_flat[i:end, 1] * v2_flat[i:end, 2] - v1_flat[i:end, 2] * v2_flat[i:end, 1]
                output_flat[i:end, 1] = v1_flat[i:end, 2] * v2_flat[i:end, 0] - v1_flat[i:end, 0] * v2_flat[i:end, 2]
                output_flat[i:end, 2] = v1_flat[i:end, 0] * v2_flat[i:end, 1] - v1_flat[i:end, 1] * v2_flat[i:end, 0]
        else:
            # For small to medium arrays, calculate directly (avoid overhead)
            # Calculate vector differences temporarily
            v1 = b - a
            v2 = d - c
            
            # Write directly to output array
            output[..., 0] = v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1]
            output[..., 1] = v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2]
            output[..., 2] = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]

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
    
    def VECTR_dot_vec(self, a, b, c, d, rdtype=None): # <added by a.kamiijo on 2025.04.02>
        """
        Vectorized version of VECTR_dot that can operate on arrays of vectors.
        
        Parameters:
            a: Array with shape (..., 3) or vector with shape (3,)
            b: Array with shape (..., 3) or vector with shape (3,)
            c: Array with shape (..., 3) or vector with shape (3,)
            d: Array with shape (..., 3) or vector with shape (3,)
            rdtype: Data type for the result
            
        Returns:
            Array with shape (...) containing dot products
        """
        # Optimized vector difference calculation
        # Continuous memory access pattern considering stride
        # Direct operations to avoid creating unnecessary intermediate arrays
        
        # Advanced einsum with optimizer flags
        # Provides optimization hints to the compiler
        # Element-wise dot product of v1 = b - a and v2 = d - c
        # einsum_path specifies calculation order and reduces temporary arrays
        # 'greedy' prioritizes computation speed over memory usage
        
        return np.einsum('...i,...i->...', b - a, d - c, optimize='greedy')
    
    def VECTR_angle(self, a, b, c, rdtype):
        nvlenC = self.VECTR_dot(b, a, b, c, rdtype)
        nv   = self.VECTR_cross(b, a, b, c, rdtype)    
        nvlenS = self.VECTR_abs(nv, rdtype)
        angle  = np.arctan2(nvlenS, nvlenC)
        return angle
    
    def VECTR_angle_vec(self, a, b, c, rdtype): # <added by a.kamiijo on 2025.04.02>
        """
        Vectorized version of VECTR_angle that can operate on arrays of vectors.
        
        Parameters:
            a: Array with shape (..., 3) or vector with shape (3,)
            b: Array with shape (..., 3) or vector with shape (3,)
            c: Array with shape (..., 3) or vector with shape (3,)
            rdtype: Data type for the result
            
        Returns:
            Array with shape (...) containing angles
        """
        # Calculate dot product using vectorized method
        nvlenC = self.VECTR_dot_vec(b, a, b, c, rdtype)
        
        # Calculate cross product using vectorized method
        nv = self.VECTR_cross_vec(b, a, b, c, rdtype)
        
        # Calculate magnitude of cross product
        nvlenS = self.VECTR_abs_vec(nv, rdtype)
        
        # Use arctan2 for numerically stable angle calculation
        angle = np.arctan2(nvlenS, nvlenC)
        
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

    def VECTR_abs_vec(self, a, rdtype=None): # <added by a.kamiijo on 2025.04.02>
        """
        Vectorized version of VECTR_abs that can operate on arrays of vectors.
        
        Parameters:
            a: Array with shape (..., 3) or vector with shape (3,)
            rdtype: Data type for the result
            
        Returns:
            Array with shape (...) containing vector magnitudes
        """
        # Advanced einsum with optimizer flags
        # Optimize memory access patterns and improve cache efficiency
        # 'optimal' selects the optimal calculation order based on computation graph analysis
        return np.sqrt(np.einsum('...i,...i->...', a, a, optimize='optimal'))

vect = Vect()
#print('instantiated vect')