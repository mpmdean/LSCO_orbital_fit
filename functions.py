import edrixs
import numpy as np
v_noccu = 9
info = edrixs.utils.get_atom_data('Cu', '3d', v_noccu, edge='L3')


I0 = 1 # scaling. Could have one number of all spectra or vary
thin = 0
Ez2 = -1.7     # -0.3 -3 variable
Exzyz = -2.12  # -0.3 -3 variable
Exy = -1.8   # -0.3 -3
tth = np.deg2rad(130) #fixed
gamma_f = 0.1 #0.03 -0.3 variable
gamma_c = info['gamma_c'] # fixed

v_soc_i = info['v_soc_i'][0] # fixed (could try varying 60 to 100%)
v_soc_n = info['v_soc_n'][0] # fixed (could try varying 60 to 100%)
c_soc = info['c_soc']        # fixed (could try varying 60 to 100%)



def get_data(eloss, filename):
    _x, _y = np.loadtxt(filename, unpack=True, skiprows=1)
    _x *= -1
    order = np.argsort(_x)
    y = np.interp(eloss, _x[order], _y[order])
    return y


def make_rixs(eloss, I0=I0, thin=thin, tth=tth, Ez2=Ez2, Exzyz=Exzyz, Exy=Exy, gamma_f=gamma_f,
             v_soc_i=v_soc_i, v_soc_n=v_soc_n, c_soc=c_soc):
    v_cfmat = make_v_cfmat(Ez2=Ez2, Exzyz=Exzyz, Exy=Exy)
    shell_name = ('d', 'p32')
    
    temperature = 25  # in K
    pol_type_rixs = [('linear', 0, 'linear', 0), ('linear', 0, 'linear', np.pi/2)]
    thout = tth - thin

    out = edrixs.ed_1v1c_py(shell_name, v_soc=(v_soc_i, v_soc_n),
                            v_cfmat=v_cfmat, c_soc=c_soc, v_noccu=v_noccu)
    eval_i, eval_n, trans_op = out
 
    rixs = edrixs.rixs_1v1c_py(
        eval_i, eval_n, trans_op, [0], eloss,
        gamma_c=gamma_c, gamma_f=gamma_f,
        thin=np.deg2rad(thin), thout=np.deg2rad(thout),
        pol_type=pol_type_rixs, gs_list=[0, 1],
        temperature=temperature
    )

    normalization_factor = np.abs(np.sin(np.deg2rad(thout))) # correct for self absorption
    spectrum = I0*rixs.sum((0, -1))/normalization_factor
    return spectrum


def make_v_cfmat(Ez2=-1.7, Exzyz=-2.12, Exy=-1.8):
    cf_rhb = np.zeros((10, 10), dtype=complex)
    orbitals = [e for e in [Ez2, Exzyz, Exzyz, 0, Exy]
                for _ in range(2)]
    cf_rhb[np.arange(10), np.arange(10)] = orbitals
    v_cfmat = edrixs.cb_op(cf_rhb, edrixs.tmat_r2c('d', True))
    return v_cfmat