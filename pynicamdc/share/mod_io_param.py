class IOParam:
    # character length
    IO_HSHORT = 16  # character length for short var.
    IO_HMID = 64  # character length for middle var.
    IO_HLONG = 256  # character length for long var.

    # data type
    IO_REAL4 = 0  # ID for 4byte real
    IO_REAL8 = 1  # ID for 8byte real
    IO_INTEGER4 = 2  # ID for 4byte int
    IO_INTEGER8 = 3  # ID for 8byte int

    # data endian
    IO_UNKNOWN_ENDIAN = 0  # ID for unknown endian
    IO_LITTLE_ENDIAN = 1  # ID for little endian
    IO_BIG_ENDIAN = 2  # ID for big endian

    # topology
    IO_ICOSAHEDRON = 0  # ID for ico grid
    IO_IGA_LCP = 1  # ID for LCP grid
    IO_IGA_MLCP = 2  # ID for MLCP grid

    # file mode (partial or complete)
    IO_SPLIT_FILE = 0  # ID for split(partical) file
    IO_INTEG_FILE = 1  # ID for integrated(complete) file

    # processor type
    IO_SINGLE_PROC = 0  # ID for single processor
    IO_MULTI_PROC = 1  # ID for multi processor

    # action type
    IO_FREAD = 0  # ID for read file
    IO_FWRITE = 1  # ID for write file
    IO_FAPPEND = 2  # ID for append file

    # data dump type
    IO_DUMP_OFF = 0  # Dumping off
    IO_DUMP_HEADER = 1  # Dump header only
    IO_DUMP_ALL = 2  # Dump all
    IO_DUMP_ALL_MORE = 3  # Dump all and more

    IO_preclist = [4, 8, 4, 8]
