"""
Minimal reader/writer for the NICAM "fio" (PaNDa) packed-binary restart format
(input/output_io_mode = "ADVANCED").

The on-disk layout is a big-endian C stream (no Fortran record markers):

  file header : header(64) note(256) + 6*uint32 (fmode,endian,topo,glevel,
                rlevel,num_of_rgn) + rgnid[num_of_rgn]*uint32 + num_of_data uint32
  per data    : varname(16) description(64) unit(16) layername(16) note(256)
                + datasize uint64 + datatype uint32 + num_layer uint32 + step uint32
                + time_start uint64 + time_end uint64 + data[datasize bytes]

Each variable's raw stream is (num_of_rgn, kall, gall); we expose it transposed to
(gall, kall, num_of_rgn) = variable_array[ij, k, l], exactly what restart_input
consumes. The read side is the same parser as tools/restart2json.py.
"""
import struct
import numpy as np

DTYPE_MAP = {0: ">f4", 1: ">f8", 2: ">i4", 3: ">i8"}
RDTYPE2FIO = {np.dtype("float32"): 0, np.dtype("float64"): 1}
DESC, NOTE = 64, 256
ITEM_DESC, ITEM_UNIT, ITEM_LAYER, ITEM_NOTE = 64, 16, 16, 256


def _cstr(b, n):
    """decode a fixed-length null-padded field."""
    return b.read(n).decode(errors="ignore").strip("\x00").strip()


def _pad(s, n):
    """encode a string into a fixed-length null-padded byte field."""
    raw = s.encode()[:n]
    return raw + b"\x00" * (n - len(raw))


def fio_read(path):
    """Parse one fio file. Returns (meta dict, {varname: array[ij,k,l]} in file order)."""
    with open(path, "rb") as f:
        header = _cstr(f, DESC)
        note = _cstr(f, NOTE)
        fmode, endian, topo, glevel, rlevel, num_of_rgn = struct.unpack(">6I", f.read(24))
        rgnid = list(struct.unpack(f">{num_of_rgn}I", f.read(4 * num_of_rgn)))
        num_of_data = struct.unpack(">I", f.read(4))[0]

        gall = (2 ** (glevel - rlevel) + 2) ** 2
        meta = dict(header=header, note=note, fmode=fmode, endian=endian, topo=topo,
                    glevel=glevel, rlevel=rlevel, num_of_rgn=num_of_rgn, rgnid=rgnid,
                    gall=gall, items=[])
        variables = {}

        for _ in range(num_of_data):
            varname = _cstr(f, 16)
            description = _cstr(f, ITEM_DESC)
            unit = _cstr(f, ITEM_UNIT)
            layername = _cstr(f, ITEM_LAYER)
            f.read(ITEM_NOTE)                                    # per-var note (unused)
            datasize, datatype, num_layer, step = struct.unpack(">Q3I", f.read(8 + 12))
            time_start, time_end = struct.unpack(">QQ", f.read(16))
            raw = f.read(datasize)
            fmt = DTYPE_MAP.get(datatype)
            if fmt is None:
                continue
            arr = np.frombuffer(raw, dtype=np.dtype(fmt))
            per = num_of_rgn * gall
            if arr.size % per != 0:
                raise ValueError(f"{path} var '{varname}': {arr.size} not divisible by "
                                 f"num_of_rgn*gall={num_of_rgn}*{gall}")
            kall = arr.size // per
            arr = arr.reshape(num_of_rgn, kall, gall).transpose(2, 1, 0)   # (ij,k,region)
            # NICAM keeps two dummy vertical levels (k=0 below surface, k=kall-1 above
            # top). nicamdc never initialises the top one -> NaN/garbage there; pyNICAM
            # builds its pole arrays from the ingested field before the vertical BC runs,
            # so garbage seeds NaN at the pole corners. The dummy levels carry no physical
            # state (BNDCND recomputes them every step); copy the adjacent physical level
            # to keep them finite. Only rewrite non-finite cells. (same as restart2json.)
            if kall >= 3:
                arr = np.array(arr)
                for kd, ksrc in ((-1, -2), (0, 1)):
                    bad = ~np.isfinite(arr[:, kd, :])
                    if bad.any():
                        arr[:, kd, :] = np.where(bad, arr[:, ksrc, :], arr[:, kd, :])
            variables[varname] = arr
            meta["items"].append(dict(varname=varname, description=description, unit=unit,
                                      layername=layername, datatype=datatype,
                                      num_layer=num_layer, step=step,
                                      time_start=time_start, time_end=time_end))
        return meta, variables


def fio_write(path, meta, items):
    """Write a fio file. `meta` supplies the file-header fields; `items` is a list of
    dicts: varname, description, unit, layername, datatype, num_layer, step, time_start,
    time_end, data (array[ij,k,l] = (gall,kall,num_of_rgn))."""
    with open(path, "wb") as f:
        f.write(_pad(meta["header"], DESC))
        f.write(_pad(meta.get("note", ""), NOTE))
        f.write(struct.pack(">6I", meta["fmode"], meta["endian"], meta["topo"],
                            meta["glevel"], meta["rlevel"], meta["num_of_rgn"]))
        f.write(struct.pack(f">{meta['num_of_rgn']}I", *meta["rgnid"]))
        f.write(struct.pack(">I", len(items)))
        for it in items:
            data = np.asarray(it["data"])
            fmt = DTYPE_MAP[it["datatype"]]
            # (ij,k,region) -> (region,k,ij) raw stream, big-endian
            raw = np.ascontiguousarray(data.transpose(2, 1, 0)).astype(fmt).tobytes()
            f.write(_pad(it["varname"], 16))
            f.write(_pad(it["description"], ITEM_DESC))
            f.write(_pad(it["unit"], ITEM_UNIT))
            f.write(_pad(it["layername"], ITEM_LAYER))
            f.write(_pad(it.get("note", ""), ITEM_NOTE))
            f.write(struct.pack(">Q3I", len(raw), it["datatype"], it["num_layer"], it["step"]))
            f.write(struct.pack(">QQ", it["time_start"], it["time_end"]))
            f.write(raw)
