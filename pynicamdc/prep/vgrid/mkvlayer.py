import argparse
import json
import toml
import numpy as np

class mkvlayer:
    """Faithful port of nicamdc prg_mkvlayer.f90.

    Arrays are 0-based with one dummy cell at each end (Fortran kdum=1):
    python position p corresponds to Fortran index p+1. Where the Fortran
    formula uses the raw 1-based loop index (ULLRICH14), the port uses p+1.
    """

    def __init__(self, num_of_layer=10, layer_type='ULLRICH14', ztop=1.E4,
                 infname='infile', outfname='outfile'):
        self.num_of_layer = num_of_layer
        self.layer_type = layer_type
        self.ztop = ztop
        self.infname = infname
        self.outfname = outfname

        self.kmin = 1
        self.kmax = num_of_layer
        self.kall = num_of_layer + 2

        self.z_c = np.zeros(self.kall)
        self.z_h = np.zeros(self.kall)

    def mk_layer_ullrich14(self):
        mu = 15.0
        for k in range(self.kmin, self.kmax + 2):
            # Fortran: fact = (real(k)/num)**2 with 1-based k = python k+1
            fact = ((k + 1) / self.num_of_layer) ** 2
            self.z_h[k] = self.ztop * (np.sqrt(mu * fact + 1.0) - 1.0) \
                                    / (np.sqrt(mu + 1.0) - 1.0)

    def mk_layer_even(self):
        dz = self.ztop / (self.kmax - self.kmin + 1)
        for k in range(self.kall):
            self.z_h[k] = dz * k

    def mk_layer_given(self):
        with open(self.infname, 'r') as f:
            lines = f.readlines()
        num_of_layer0 = int(lines[0].strip())
        if num_of_layer0 != self.num_of_layer:
            print(f"Mismatch num_of_layer (input,request) = {num_of_layer0}, {self.num_of_layer}")
        self.z_h[self.kmin:self.kmax + 2] = np.array([float(x.strip()) for x in lines[1:]])

    def output_layer(self):
        """pyNICAM vgrid JSON: set1 -> GRD_gz (centers), set2 -> GRD_gzh (half levels)."""
        with open(self.outfname, 'w') as f:
            json.dump({"set1": self.z_c.tolist(), "set2": self.z_h.tolist()}, f)
        print(f"wrote {self.outfname}  (z{self.num_of_layer}, top center = {self.z_c[-2]:.1f} m)")

    def generate_layers(self):
        if self.layer_type == 'ULLRICH14':
            self.mk_layer_ullrich14()
        elif self.layer_type == 'EVEN':
            self.mk_layer_even()
        elif self.layer_type == 'GIVEN':
            self.mk_layer_given()
        else:
            raise Exception("Unknown layer type.")

        self.z_h[self.kmin - 1] = self.z_h[self.kmin] - (self.z_h[self.kmin + 1] - self.z_h[self.kmin])

        for k in range(self.kmin - 1, self.kmax + 1):
            self.z_c[k] = self.z_h[k] + 0.5 * (self.z_h[k + 1] - self.z_h[k])
        self.z_c[self.kmax + 1] = self.z_h[self.kmax + 1] + 0.5 * (self.z_h[self.kmax + 1] - self.z_h[self.kmax])

        self.output_layer()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate pyNICAM vertical-grid JSON ({set1: gz, set2: gzh})")
    ap.add_argument("--config", default='../../case/config/mkvlayer.toml',
                    help="config toml with [mkvlayer] num_of_layer/layer_type/ztop/infname/outfname")
    args = ap.parse_args()

    cnfs = toml.load(args.config)['mkvlayer']

    layer = mkvlayer(num_of_layer=cnfs['num_of_layer'],
                     layer_type=cnfs['layer_type'],
                     ztop=cnfs.get('ztop', 1.E4),
                     infname=cnfs.get('infname', 'infile'),
                     outfname=cnfs['outfname'])

    layer.generate_layers()
