import numpy as np

class mkvlayer:
    def __init__(self, num_of_layer = 10, layer_type = 'ULLRICH14', ztop = 1.E4, infname = 'infile', outfname = 'outfile'):
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
        for k in range(self.kmin, self.kmax+2):
            fact = ( k / self.num_of_layer )**2
            self.z_h[k] =  self.ztop * ( np.sqrt( mu * fact + 1.0 ) - 1.0 ) / ( np.sqrt( mu + 1.0 ) - 1.0 )

    def mk_layer_even(self):
        dz = self.ztop / (self.kmax-self.kmin+1)
        for k in range(self.kall):
            self.z_h[k] = dz * k

    def mk_layer_given(self):
        with open(self.infname, 'r') as f:
            lines = f.readlines()
        num_of_layer0 = int(lines[0].strip())
        if num_of_layer0 != self.num_of_layer:
            print(f"Mismach num_of_layer (input,request) = {num_of_layer0}, {self.num_of_layer}")
        self.z_h[self.kmin:self.kmax+2] = np.array([float(x.strip()) for x in lines[1:]])

    def output_layer(self):
        with open(self.outfname, 'w') as f:
            f.write(f"{self.num_of_layer}\n")
            f.write(' '.join(map(str, self.z_c))+'\n')
            f.write(' '.join(map(str, self.z_h))+'\n')

    def generate_layers(self):
        if self.layer_type == 'ULLRICH14':
            self.mk_layer_ullrich14()
        elif self.layer_type == 'EVEN':
            self.mk_layer_even()
        elif self.layer_type == 'GIVEN':
            self.mk_layer_given()
        else:
            raise Exception("Unknown layer type.")

        self.z_h[self.kmin-1] = self.z_h[self.kmin] - ( self.z_h[self.kmin+1] - self.z_h[self.kmin] )

        for k in range(self.kmin-1, self.kmax+1):
            self.z_c[k] = self.z_h[k] + 0.5 * ( self.z_h[k+1] - self.z_h[k] )
        self.z_c[self.kmax+1] = self.z_h[self.kmax+1] + 0.5 * ( self.z_h[self.kmax+1] - self.z_h[self.kmax] )

        self.output_layer()

# Example usage
layer = mkvlayer(num_of_layer=int(config_params['num_of_layer']), 
                 layer_type=config_params['layer_type'], 
                 infname="/mnt/data/_vgrid_40L_exp.dat", 
                 outfname='/mnt/data/output.txt')
layer.generate_layers()
