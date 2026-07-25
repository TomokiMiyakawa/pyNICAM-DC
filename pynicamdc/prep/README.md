# prep — self-contained input generation for pyNICAM-DC

Everything the model needs for a NEW glevel / rlevel / pe decomposition can be
generated here, with no nicamdc binaries or legacy grid files:

| input | tool | validation |
|---|---|---|
| mnginfo toml (region topology + rank map) | `mnginfo/mkmnginfo.py` | `validate_mkmnginfo.py`: regenerates all 15 previously validated tomls (rl00/rl01/rl03), 15/15 match |
| vertical grid JSON `{set1: gz, set2: gzh}` | `vgrid/mkvlayer.py` | `validate_mkvlayer.py`: BIT-EXACT vs the Fortran nicamdc mkvlayer binary for ULLRICH14 / EVEN / GIVEN (`vgrid/reference/*.dat`) |
| horizontal grid boundary npz | `hgrid/mkrawgrid.py` | `validate_hgrid.py`: generated gl05rl00 vs the nicamdc-provenance tutorial grid, max point offset 1.4e-13; tier2 JW golden PASS end-to-end |

## Making a new input set

Run each tool from its own directory (or pass `--config` with your own toml):

```bash
cd prep/mnginfo
python mkmnginfo.py --config my.toml      # [mkmnginfo] rlevel / prc_num / output_fname

cd ../vgrid
python mkvlayer.py --config my.toml       # [mkvlayer] num_of_layer / layer_type
                                          #   (ULLRICH14|EVEN|GIVEN) / ztop / infname / outfname

cd ../hgrid
python mkrawgrid.py --comm serial         # 1 rank (login-node safe; rl00 = 10 regions)
mpirun -n N python mkrawgrid.py --comm mpi  # pe>1: [rgnmngparam] mnginfo must match N
```

`mkrawgrid.py` runs the full chain MKGRD_standard → spring (vectorized; the
original scalar loop is kept under `--spring-loop` and is bit-identical) →
gravcenter (fills GRD_xt) and writes the per-rank boundary npz that
`GRD_input_hgrid` reads with `hgrid_io_mode = "npz"`. The pole grid is NOT
stored — the model regenerates it (`GRD_gen_plgrid`).

Point the model at the products:

```toml
[grdparam]
hgrid_io_mode = "npz"
hgrid_fname = ".../boundary_GL05RL00.pe"   # + <rank:08d>.npz
vgrid_fname = ".../vgrid30_ullrich14.json"
```

## Caveats

- **Frozen historical grids stay frozen.** Python-generated grids reproduce the
  nicamdc algorithm to ~1e-13 but are not bit-identical to the legacy files;
  validated benchmark/golden series keep using their original boundary npz.
- **Historical vgrids** (e.g. `vgrid30_400m_dcmip`) use an older half-level
  convention (`gzh[kmin]=0`) that the current mkvlayer EVEN cannot produce for
  any ztop. Regenerate them only via GIVEN from their own half levels.
- prerotate / stretch / shrink / rotate (reduced-planet options) are not
  ported; the config flags exist but must stay false.
- Legacy panNICAM `grid.rgn` decoding (for grids that predate this tool) lives
  in the benchmark kit's `build_hires_inputs.py`; it is the alternative input
  path for gl10/gl11 where the frozen decoded npz are already validated.
