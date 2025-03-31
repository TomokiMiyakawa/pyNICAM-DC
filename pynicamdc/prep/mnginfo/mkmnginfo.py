import numpy as np
import toml
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
share_module_dir = os.path.join(script_dir, "../../share")  
sys.path.insert(0, share_module_dir)

from mod_adm import adm

class Mkmnginfo:
    def __init__(self):
        #self.adm = Adm()
        
        # Load configurations from TOML file
        cnfs = toml.load('../../case/config/mkmnginfo.toml')['mkmnginfo']
        self.rlevel = cnfs['rlevel']
        self.prc_num = cnfs['prc_num']
        self.output_fname = cnfs['output_fname']

        print('prc_num', self.prc_num)

        self.nmax_dmd = 10

        # Define and initialize the matrix  zero-based region connection for Icosahedron
        # Order: [SW,NW,NE,SW] 
        matrix_init = np.array([
            [5, 4, 1, 9],
            [9, 0, 2, 8],
            [8, 1, 3, 7],
            [7, 2, 4, 6],
            [6, 3, 0, 5],
            [6, 4, 0, 9],
            [7, 3, 4, 5],
            [8, 2, 3, 6],
            [9, 1, 2, 7],
            [5, 0, 1, 8]
        ])

        # zero-based indexing 
        #                                3+1 =4            10
        self.dmd_data = np.zeros((adm.I_SE + 1, self.nmax_dmd), dtype=int)
        for j in range(self.nmax_dmd):    # 0-9 because nmax_dmd=10
            for i in range(adm.I_SE + 1):  # 0-3 because I_SE=3
                self.dmd_data[i, j] = matrix_init[j, i]

                
    def generate_mngtab(self, rl, nmax_prc, fname):
        rgnlen = 2 ** rl
        all_rgn = self.nmax_dmd * rgnlen * rgnlen
        n_mng = int(all_rgn/nmax_prc)
        print('rl', rl)
        print('nmax_prc', nmax_prc)
        print('all_rgn', all_rgn)

        # Base structure of the output toml data
        data = {
            "Title": "mnginfo for rlevel= " + str(rl) + ", process number= " + str(nmax_prc) ,
            "RGN_INFO": {
                "NUM_OF_RGN": all_rgn
            },
            "PROC_INFO": {
                "NUM_OF_PROC": nmax_prc,
                "NUM_OF_MNG": n_mng
            },
            "RGN_LINK_INFO": {},
            "RGN_MNG_INFO": {}
        }
                
        # Use numpy to create the 3D array (much faster and more efficient)
        #                           1 + 1 = 2,      3 + 1 = 4
        rgn_tab = np.zeros((adm.I_DIR + 1, adm.I_SE + 1, all_rgn), dtype=int)

        # You can use numpy's saving functions if you need to store the array
        # np.save(fname, rgn_tab)  # example of saving

        for d in range(self.nmax_dmd):           #  10 dyamonds  0-9
            for j in range(rgnlen):              #  rl divide x  0-1 if rgnlen=2 (rl=1)
                for i in range(rgnlen):          #  rl divide y  0-1 if rgnlen=2 (rl=1)
                    l = (rgnlen * rgnlen) * d + rgnlen * j + i   #  'l' is a 0-based index counting all regions
                    #                   0                3 + 1               
                    #for k in range(adm.I_SW, adm.I_SE + 1):  # Loop over the range of directions
                    #                     0-3
                    for k in range(adm.I_SE + 1):  # Loop over the range of directions
                        if k == adm.I_SW:
                            if j == 0:
                                if d < 5:
                                    i_nb = i
                                    j_nb = rgnlen - 1
                                    d_nb = self.dmd_data[k, d]
                                    edgid_nb = adm.I_NE
                                else:
                                    i_nb = rgnlen - 1
                                    j_nb = rgnlen - i - 1
                                    d_nb = self.dmd_data[k, d]
                                    edgid_nb = adm.I_SE
                            else:
                                i_nb = i
                                j_nb = j - 1
                                d_nb = d   # Same domain  
                                edgid_nb = adm.I_NE
                        
                        elif k == adm.I_NW:
                            if i == 0:  # If we are at the 'west' edge of the region
                                if d < 5:
                                    i_nb = rgnlen - j - 1  # Reflect for the opposite edge
                                    j_nb = rgnlen - 1
                                    d_nb = self.dmd_data[k, d]  # Direction based on dmd_data
                                    edgid_nb = adm.I_NE  # Edge ID for the north-east
                                else:
                                    i_nb = rgnlen - 1
                                    j_nb = j
                                    d_nb = self.dmd_data[k, d]  # Direction based on dmd_data
                                    edgid_nb = adm.I_SE  # Edge ID for the south-east
                            else:
                                i_nb = i - 1
                                j_nb = j
                                d_nb = d  # Same domain
                                edgid_nb = adm.I_SE  # Edge ID for the south-east
                                
                        elif k == adm.I_NE:
                            if j == rgnlen - 1:  # If we are at the 'north' edge of the region
                                if d < 5:
                                    i_nb = 0  # Start at the beginning of the i-index
                                    j_nb = rgnlen - i - 1  # Reflect for the opposite edge
                                    d_nb = self.dmd_data[k, d]  # Direction based on dmd_data
                                    edgid_nb = adm.I_NW  # Edge ID for the north-west
                                else:
                                    i_nb = i
                                    j_nb = 0  # Start at the beginning of the j-index
                                    d_nb = self.dmd_data[k, d]  # Direction based on dmd_data
                                    edgid_nb = adm.I_SW  # Edge ID for the south-west
                            else:
                                i_nb = i
                                j_nb = j + 1
                                d_nb = d   # Same domain
                                edgid_nb = adm.I_SW  # Edge ID for the south-west
                        
                        elif k == adm.I_SE:
                            if i == rgnlen - 1:  # If we are at the 'east' edge of the region
                                if d < 5:
                                    i_nb = 0  # Start at the beginning of the i-index
                                    j_nb = j  # Stay in the same row
                                    d_nb = self.dmd_data[k, d]  # Direction based on dmd_data
                                    edgid_nb = adm.I_NW  # Edge ID for the north-west
                                else:
                                    i_nb = rgnlen - j - 1  # Reflect for the opposite edge
                                    j_nb = 0  # Start at the beginning of the j-index
                                    d_nb = self.dmd_data[k, d]  # Direction based on dmd_data
                                    edgid_nb = adm.I_SW  # Edge ID for the south-west
                            else:
                                i_nb = i + 1
                                j_nb = j
                                d_nb = d  # Same domain
                                edgid_nb = adm.I_NW  # Edge ID for the north-west

                                
                        l_nb = (rgnlen * rgnlen) * (d_nb) + rgnlen * (j_nb) + i_nb  # Adjusted index
                        rgn_tab[adm.I_RGNID][k][l] = l_nb  # Adjusted for 0-based indexing
                        rgn_tab[adm.I_DIR][k][l] = edgid_nb  # Adjusted for 0-based indexing

                        data["RGN_LINK_INFO"][f"{l:06}"] = {
                            "RGNID": l,
                            "SW": [ int(rgn_tab[adm.I_RGNID][adm.I_SW][l]) ,
                                    int(rgn_tab[adm.I_DIR][adm.I_SW][l])
                                   ] ,
                            "NW": [ int(rgn_tab[adm.I_RGNID][adm.I_NW][l]) ,
                                    int(rgn_tab[adm.I_DIR][adm.I_NW][l])
                                   ],
                            "NE": [ int(rgn_tab[adm.I_RGNID][adm.I_NE][l]) ,
                                    int(rgn_tab[adm.I_DIR][adm.I_NE][l])
                                   ],
                            "SE": [ int(rgn_tab[adm.I_RGNID][adm.I_SE][l]) ,
                                    int(rgn_tab[adm.I_DIR][adm.I_SE][l])
                                   ],
                        }

        for peid in range(data["PROC_INFO"]["NUM_OF_PROC"]):
            data["RGN_MNG_INFO"][f"{peid:06}"] = {
                "PEID": peid,
                "NUM_OF_MNG": n_mng,
                "MNG_RGNID": [j for j in range( peid*n_mng, (peid+1)*n_mng)]
            }

        # Convert data to TOML format
        toml_file_content = toml.dumps(data)

        # Write the TOML content to a file
        with open(fname, 'w') as file:
            file.write(toml_file_content)
            print(f"TOML file created at {fname}")

        pass

mk=Mkmnginfo()
mk.generate_mngtab(mk.rlevel,mk.prc_num,mk.output_fname)

