from os.path import exists, join
from os import getenv
import numpy as np
from lib import magnet_simulations
from time import time, sleep
from muon_slabs import initialize
import json
import h5py
from snoopy import RacetrackCoil

RESOL_DEF = magnet_simulations.RESOL_DEF
MATERIALS_DIR = 'data/materials'
Z_GAP = 10 # in cm
SC_Ymgap = magnet_simulations.SC_Ymgap
N_PARAMS = 15
SHIFT = -214
CAVERN_TRANSITION = 2051.8+SHIFT

def get_field(resimulate_fields = False,
            params = None,
            file_name = None,
            only_grid_params = False,
            **kwargs_field):
    '''Returns the field map for the given parameters. If from_file is True, the field map is loaded from the file_name.'''
    if resimulate_fields:
        d_space = kwargs_field['d_space']
        fields = magnet_simulations.simulate_field(params, file_name = file_name,**kwargs_field)['B']
    elif exists(file_name):
        print('Using field map from file', file_name)
        with h5py.File(file_name, 'r') as f:
            fields = f["B"][:]
            d_space = f["d_space"][:].tolist()
    if only_grid_params: 
        fields = {'B': file_name if file_name is not None else fields,
                'range_x': [d_space[0][0],d_space[0][1], RESOL_DEF[0]],
                'range_y': [d_space[1][0],d_space[1][1], RESOL_DEF[1]],
                'range_z': [d_space[2][0],d_space[2][1], RESOL_DEF[2]]}
    return fields


def CreateArb8(arbName, medium, dZ, corners, magField, field_profile,
               tShield, z_translation):
    tShield['components'].append({
        'corners' : (corners/100).tolist(),
        'field_profile' : field_profile,
        'field' : magField,
        'name': arbName,
        'dz' : float(dZ)/100,
        "z_center" : float(z_translation)/100,
        "material": medium
    })


def create_magnet(magnetName, medium, tShield,
                  fields,field_profile, dX,
                  dY, dX2, dY2, dZ, middleGap,
                  middleGap2,ratio_yoke_1, ratio_yoke_2, dY_yoke_1,dY_yoke_2, gap,
                  gap2, Z, Ymgap = 0):
    dY += Ymgap #by doing in this way, the gap is filled with iron in Geant4, but simplifies
    coil_gap = gap
    coil_gap2 = gap2
    anti_overlap = 0.01


    cornersMainL = np.array([
        middleGap, 
        -(dY +dY_yoke_1)- anti_overlap, 
        middleGap, 
        dY + dY_yoke_1- anti_overlap,
        dX + middleGap, 
        dY- anti_overlap, 
        dX + middleGap,
        -(dY- anti_overlap),
        middleGap2,
        -(dY2 + dY_yoke_2- anti_overlap), middleGap2, 
        dY2 + dY_yoke_2- anti_overlap,
        dX2 + middleGap2, 
        dY2- anti_overlap, 
        dX2 + middleGap2,
        -(dY2- anti_overlap)])

    cornersTL = np.array((middleGap + dX,dY,
                            middleGap,
                            dY + dY_yoke_1,
                            dX + ratio_yoke_1*dX + middleGap + coil_gap,
                            dY + dY_yoke_1,
                            dX + middleGap + coil_gap,
                            dY,
                            middleGap2 + dX2,
                            dY2,
                            middleGap2,
                            dY2 + dY_yoke_2,
                            dX2 + ratio_yoke_2*dX2 + middleGap2 + coil_gap2,
                            dY2 + dY_yoke_2,
                            dX2 + middleGap2 + coil_gap2,
                            dY2))

    cornersMainSideL = np.array((dX + middleGap + gap,
                                 -(dY), 
                                 dX + middleGap + gap,
                                dY, 
                                dX + ratio_yoke_1*dX + middleGap + gap, 
                                dY + dY_yoke_1,
                                dX + ratio_yoke_1*dX + middleGap + gap, 
                                -(dY + dY_yoke_1), 
                                dX2 + middleGap2 + gap2,
                                -(dY2), 
                                dX2 + middleGap2 + gap2, 
                                dY2,
                                dX2 + ratio_yoke_2*dX2 + middleGap2 + gap2, 
                                dY2 + dY_yoke_2, 
                                dX2 + ratio_yoke_2*dX2 + middleGap2 + gap2,
                                -(dY2 + dY_yoke_2)))

    
    cornersMainR = np.zeros(16, np.float16)
    cornersCLBA = np.zeros(16, np.float16)
    cornersMainSideR = np.zeros(16, np.float16)
    cornersCLTA = np.zeros(16, np.float16)
    cornersCRBA = np.zeros(16, np.float16)
    cornersCRTA = np.zeros(16, np.float16)

    cornersTR = np.zeros(16, np.float16)
    cornersBL = np.zeros(16, np.float16)
    cornersBR = np.zeros(16, np.float16)


    # Use symmetries to define remaining magnets
    for i in range(16):
        cornersMainR[i] = -cornersMainL[i]
        cornersMainSideR[i] = -cornersMainSideL[i]
        cornersCRTA[i] = -cornersCLBA[i]
        cornersBR[i] = -cornersTL[i]

    # Need to change order as corners need to be defined clockwise
    for i in range(8):
        j = (11 - i) % 8
        cornersCLTA[2 * j] = cornersCLBA[2 * i]
        cornersCLTA[2 * j + 1] = -cornersCLBA[2 * i + 1]
        cornersTR[2 * j] = -cornersTL[2 * i]
        cornersTR[2 * j + 1] = cornersTL[2 * i + 1]

    for i in range(16):
        cornersCRBA[i] = -cornersCLTA[i]
        cornersBL[i] = -cornersTR[i]

    str1L = "_MiddleMagL"
    str1R = "_MiddleMagR"
    str2 = "_MagRetL"
    str3 = "_MagRetR"
    str4 = "_MagCLB"
    str5 = "_MagCLT"
    str6 = "_MagCRT"
    str7 = "_MagCRB"
    str8 = "_MagTopLeft"
    str9 = "_MagTopRight"
    str10 = "_MagBotLeft"
    str11 = "_MagBotRight"

    theMagnet = {
        'components' : [],
        'dz' : float(dZ) / 100,
        'z_center' : float(Z) / 100,
        'material' : medium,
    }

    if field_profile == 'uniform':
        CreateArb8(magnetName + str1L, medium, dZ, cornersMainL, fields[0], field_profile, theMagnet, Z)
        CreateArb8(magnetName + str1R, medium, dZ, cornersMainR, fields[0], field_profile, theMagnet, Z)
        CreateArb8(magnetName + str2, medium, dZ, cornersMainSideL, fields[1], field_profile, theMagnet, Z)
        CreateArb8(magnetName + str3, medium, dZ, cornersMainSideR, fields[1], field_profile, theMagnet, Z)
        CreateArb8(magnetName + str8, medium, dZ, cornersTL, fields[3], field_profile, theMagnet, Z)
        CreateArb8(magnetName + str9, medium, dZ, cornersTR, fields[2], field_profile, theMagnet, Z)
        CreateArb8(magnetName + str10, medium, dZ, cornersBL, fields[2], field_profile, theMagnet, Z)
        CreateArb8(magnetName + str11, medium, dZ, cornersBR, fields[3], field_profile, theMagnet, Z)

    else:
        CreateArb8(magnetName + str1L, medium, dZ, cornersMainL, fields, field_profile, theMagnet, Z)
        CreateArb8(magnetName + str8, medium, dZ, cornersTL, fields, field_profile, theMagnet, Z)
        CreateArb8(magnetName + str2, medium, dZ, cornersMainSideL, fields, field_profile, theMagnet, Z)
        CreateArb8(magnetName + str1R, medium, dZ, cornersMainR, fields, field_profile, theMagnet, Z)
        CreateArb8(magnetName + str3, medium, dZ, cornersMainSideR, fields, field_profile, theMagnet, Z)
        CreateArb8(magnetName + str9, medium, dZ, cornersTR, fields, field_profile, theMagnet, Z)
        CreateArb8(magnetName + str10, medium, dZ, cornersBL, fields, field_profile, theMagnet, Z)
        CreateArb8(magnetName + str11, medium, dZ, cornersBR, fields, field_profile, theMagnet, Z)


    tShield['magnets'].append(theMagnet)


def design_muon_shield(params,fSC_mag = True, simulate_fields = False, field_map_file = None, cores_field:int = 1, NI_from_B = True, use_diluted = False, SND = False):

    n_magnets = len(params)
    length = (params[:,:0].sum() + 2*params[:,1].sum()).item()

    tShield = {
        'dz': length / 100,
        'magnets':[],
        'global_field_map': {},
    }

    Z = 0
    cost = 0
    max_x = 0
    max_y = 0
    for nM,magnet in enumerate(params):
        magnet = magnet.tolist()
        zgap = magnet[0]
        dZ = magnet[1]
        dXIn = magnet[2]
        dXOut = magnet[3]
        dYIn = magnet[4]
        dYOut = magnet[5]
        gapIn = magnet[6]
        gapOut = magnet[7]
        ratio_yokesIn = magnet[8]
        ratio_yokesOut = magnet[9]
        dY_yokeIn = magnet[10]
        dY_yokeOut = magnet[11]
        midGapIn = magnet[12]
        midGapOut = magnet[13]
        NI = magnet[14]

        if dZ < 1 or dXIn < 1: Z += dZ + zgap; continue

        SC_threshold = 3.0 if NI_from_B else 1e6
        is_SC = fSC_mag and (abs(NI)>SC_threshold)
        Ymgap = SC_Ymgap if is_SC else 0
        Z += zgap + dZ

        if simulate_fields or field_map_file is not None:
            field_profile = 'global'
            fields_s = [[],[],[]]
        else:
            field_profile = 'uniform'
            ironField_s = 5.7 if is_SC else 1.9
            if NI<0:
                ironField_s = -ironField_s
            magFieldIron_s = [0., ironField_s, 0.]
            RetField_s = [0., -ironField_s/ratio_yokesIn, 0.]
            ConRField_s = [-ironField_s/ratio_yokesIn, 0., 0.]
            ConLField_s = [ironField_s/ratio_yokesIn, 0., 0.]
            fields_s = [magFieldIron_s, RetField_s, ConRField_s, ConLField_s]

        create_magnet(f"Mag_{nM}", "G4_Fe", tShield, fields_s, field_profile, dXIn, dYIn, dXOut,
              dYOut, dZ, midGapIn, midGapOut, ratio_yokesIn, ratio_yokesOut,
              dY_yokeIn, dY_yokeOut, gapIn, gapOut, Z, Ymgap=Ymgap)
        yoke_type = 'Mag1' if NI>0 else 'Mag3'
        if is_SC: yoke_type = 'Mag2'
        cost += get_iron_cost([0,dZ, dXIn, dXOut, dYIn, dYOut, gapIn, gapOut, ratio_yokesIn, ratio_yokesOut, dY_yokeIn, dY_yokeOut, midGapIn, midGapOut], Ymgap=Ymgap)        
        #cost += estimate_electrical_cost(np.array([0,dZ, dXIn, dXOut, dYIn, dYOut, gapIn, gapOut, ratio_yokesIn, ratio_yokesOut, dY_yokeIn, dY_yokeOut, midGapIn, midGapOut, NI]), Ymgap=Ymgap, yoke_type=yoke_type, NI_from_B=NI_from_B)
        Z += dZ
        max_x = max(max_x, np.max(dXIn + dXIn * ratio_yokesIn + gapIn+midGapIn), np.max(dXOut + dXOut * ratio_yokesOut+gapOut+midGapOut))
        max_y = max(max_y, np.max(dYIn + dY_yokeIn), np.max(dYOut + dY_yokeOut))
        if SND and nM == (n_magnets - 2): 
            print("Adding SND after magnet", nM)
            if (midGapIn >= 30) and (midGapOut >= 30):
                gap = 1
                dX = 30.-gap
                dY = 30.-gap
                dZ_snd = 172/2
                Z_snd = Z-dZ_snd
                corners = np.array([
                    -dX, -dY,
                    dX, -dY,
                    dX, dY,
                    -dX, dY,
                    -dX, -dY,
                    dX, -dY,
                    dX, dY,
                    -dX, dY
                ])
                Block = {
                    'components' : [],
                    'dz' : dZ_snd / 100,
                    'z_center' : Z_snd / 100,
                }
                CreateArb8('SND_Emu_Si', 'G4_Fe', dZ_snd, corners, [0.,0.,0.], 'uniform', Block, Z_snd)
                tShield['magnets'].append(Block)
            else:
                print("WARNING")
                print("No space for the SND: midGapIn[5] <= 30 or midGapOut[5] <= 30, got", midGapIn, midGapOut)
    if field_map_file is not None or simulate_fields: 
        simulate_fields = simulate_fields or (not exists(field_map_file))
        resol = RESOL_DEF
        max_x = int((max_x.item() // resol[0]) * resol[0])
        max_y = int((max_y.item() // resol[1]) * resol[1])
        d_space = ((0,max_x+30), (0,max_y+30), (-50, int(((length+200) // resol[2]) * resol[2])))

        field_map = get_field(simulate_fields,np.asarray(params),
                              Z_init = 0, fSC_mag=fSC_mag, 
                              resol = resol, d_space = d_space,
                              file_name=field_map_file, only_grid_params=True, NI_from_B_goal = NI_from_B,
                              cores = min(cores_field,n_magnets), use_diluted = use_diluted)
        tShield['global_field_map'] = field_map

    tShield['cost'] = cost
    field_profile = 'global' if simulate_fields else 'uniform'
    return tShield


def get_design_from_params(params, 
                           fSC_mag:bool = True, 
                           force_remove_magnetic_field = False,
                           simulate_fields = False,
                           field_map_file = None,
                           sensitive_film_params:dict = {'dz': 0.01, 'dx': 4, 'dy': 6,'position':82},
                           add_cavern:bool = True,
                           cores_field:int = 1,
                           NI_from_B = True, 
                           use_diluted = False,
                           SND = False):
    params = np.round(params, 2)
    assert params.shape[-1] == 15
    shield = design_muon_shield(params, fSC_mag, simulate_fields = simulate_fields, field_map_file = field_map_file, cores_field=cores_field, NI_from_B = NI_from_B, use_diluted=use_diluted, SND=SND)
    
    World_dZ = 200
    World_dX = World_dY = 20 if add_cavern else 15
    
    if force_remove_magnetic_field:
        for mag in shield['magnets']:
            mag['z_center'] = mag['z_center']
            for x in mag['components']:
                    x['field'] = (0.0, 0.0, 0.0)
                    x['field_profile'] = 'uniform'

    shield.update({
     "worldSizeX": World_dX, "worldSizeY": World_dY, "worldSizeZ": World_dZ,
        "type" : 1,
        "limits" : {
            "max_step_length": 0.05,
            "minimum_kinetic_energy": 0.1,
        },
    })
    if sensitive_film_params is not None:
        sens_films = []
        for sens in sensitive_film_params:
            pos = sens['position']
            sens_films.append({
            "name": "SensitiveFilm_{}".format(pos),
            "z_center" : pos,
            "dz" : sens['dz'],
            "dx": sens['dx'],
            "dy": sens['dy']})
        shield.update({"sensitive_film": sens_films})
    return shield



def initialize_geant4(detector, seed = None):
    if seed is None: seeds = (np.random.randint(256), np.random.randint(256), np.random.randint(256), np.random.randint(256))
    else: seeds = (seed, seed, seed, seed)
    detector = json.dumps(detector)
    output_data = initialize(*seeds,detector)
    return output_data
