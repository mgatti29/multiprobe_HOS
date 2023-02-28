import healpy as hp
import asdf
import numpy as np
import copy
import pandas as pd
from bornraytrace import lensing as brk
from bornraytrace import intrinsic_alignments as iaa
import bornraytrace
from astropy.cosmology import FlatLambdaCDM,wCDM
from astropy import units as u
import os
from astropy.table import Table  
from astropy.cosmology import z_at_value
import astropy.io.fits as fits
import pickle

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)
        f.close()

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute =  pickle.load(f)
        f.close()
    return mute

def rotate_map_approx(mask, rot_angles, flip=False,nside = 1024):
    alpha, delta = hp.pix2ang(nside, np.arange(len(mask)))

    rot = hp.rotator.Rotator(rot=rot_angles, deg=True)
    rot_alpha, rot_delta = rot(alpha, delta)
    if not flip:
        rot_i = hp.ang2pix(nside, rot_alpha, rot_delta)
    else:
        rot_i = hp.ang2pix(nside, np.pi-rot_alpha, rot_delta)
    rot_map = mask*0.
    rot_map[rot_i] =  mask[np.arange(len(mask))]
    return rot_map

def gk_inv(K,KB,nside,lmax):

    alms = hp.map2alm(K, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsE = alms/( 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
   
    kalmsE[ell == 0] = 0.0

    
    alms = hp.map2alm(KB, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsB = alms/( 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
   
    kalmsB[ell == 0] = 0.0

    _,e1t,e2t = hp.alm2map([kalmsE,kalmsE,kalmsB] , nside=nside, lmax=lmax, pol=True)
    return e1t,e2t# ,r



def random_draw_ell_from_w(wi,w,e1,e2):
    '''
    wi: input weights
    w,e1,e2: all the weights and galaxy ellipticities of the catalog.
    e1_,e2_: output ellipticities drawn from w,e1,e2.
    '''


    ell_cont = dict()
    for w_ in np.unique(w):
        mask_ = w == w_
        w__ = np.int(w_*10000)
        ell_cont[w__] = [e1[mask_],e2[mask_]]

    e1_ = np.zeros(len(wi))
    e2_ = np.zeros(len(wi))


    for w_ in np.unique(wi):
        mask_ = (wi*10000).astype(np.int) == np.int(w_*10000)
        e1_[mask_] = ell_cont[np.int(w_*10000)][0][np.random.randint(0,len(ell_cont[np.int(w_*10000)][0]),len(e1_[mask_]))]
        e2_[mask_] = ell_cont[np.int(w_*10000)][1][np.random.randint(0,len(ell_cont[np.int(w_*10000)][0]),len(e1_[mask_]))]

    return e1_,e2_

def IndexToDeclRa(index, nside,nest= False):
    theta,phi=hp.pixelfunc.pix2ang(nside ,index,nest=nest)
    return -np.degrees(theta-np.pi/2.),np.degrees(phi)

def convert_to_pix_coord(ra, dec, nside=1024):
    """
    Converts RA,DEC to hpix coordinates
    """

    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    return pix

def apply_random_rotation(e1_in, e2_in):
    np.random.seed() # CRITICAL in multiple processes !
    rot_angle = np.random.rand(len(e1_in))*2*np.pi #no need for 2?
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = + e1_in * cos + e2_in * sin
    e2_out = - e1_in * sin + e2_in * cos
    return e1_out, e2_out

def addSourceEllipticity(self,es,es_colnames=("e1","e2"),rs_correction=True,inplace=False):

		"""

		:param es: array of intrinsic ellipticities, 

		"""

		#Safety check
		assert len(self)==len(es)

		#Compute complex source ellipticity, shear
		es_c = np.array(es[es_colnames[0]]+es[es_colnames[1]]*1j)
		g = np.array(self["shear1"] + self["shear2"]*1j)

		#Shear the intrinsic ellipticity
		e = es_c + g
		if rs_correction:
			e /= (1 + g.conjugate()*es_c)

		#Return
		if inplace:
			self["shear1"] = e.real
			self["shear2"] = e.imag
		else:
			return (e.real,e.imag)
        
        
def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048,nosh=True):
    """
    Convert shear to convergence on a sphere. In put are all healpix maps.
    """

    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!


    ell, emm = hp.Alm.getlm(lmax=lmax)
    if nosh:
        almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
        almsB = alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
    else:
        almsE = alms[1] * 1.
        almsB = alms[2] * 1. 
    almsE[ell == 0] = 0.0
    almsB[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    almsB[ell == 1] = 0.0



    almssm = [alms[0], almsE, almsB]


    kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
    E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
    B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)

    return E_map, B_map, almsE



rel = 25
path = '/global/cfs/cdirs//desi/public/cosmosim/boryanah_AbacusLensing/'

def run_it(number):
    
    if number <10:
        sim = 'AbacusSummit_base_c000_ph00{0}/'.format(number)
    else:
        sim = 'AbacusSummit_base_c000_ph0{0}/'.format(number)
        
    config = dict()
    # DES redhsift distributions
    config['2PT_FILE'] = '/global/homes/m/mgatti/Mass_Mapping/HOD/PKDGRAV_CODE//2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_6000HR.fits'  
    config['sources_bins'] = [0,1,2,3]
    #nside of the final maps
    config['nside'] = 1024


    # cosmology
    #https://abacussummit.readthedocs.io/en/latest/cosmologies.html#cosmologies-table
    config['h'] = 0.6736
    config['om'] = 0.1200/((config['h']**2))+ 0.02237/((config['h']**2))
    config['w0'] = -1.

    z_bin_edges = np.linspace(0.15,2.55,49)
    z_bounds     = dict()                                                                                         
    z_bounds['z-high'] = z_bin_edges[1:]
    z_bounds['z-low'] = z_bin_edges[:-1]



    cosmology = wCDM(H0=config['h']*100.*u.km / u.s / u.Mpc,
                 Om0=config['om'],
                 Ode0=1-config['om'],
                 w0=config['w0'] )


    kappa_pref_evaluated = brk.kappa_prefactor(cosmology.H0, cosmology.Om0, length_unit = 'Mpc')
    comoving_edges =  cosmology.comoving_distance(z_bin_edges)
    z_centre = np.array([z_at_value(cosmology.comoving_distance, 0.5*(comoving_edges[i]+comoving_edges[i+1]))  for i in range(len(comoving_edges)-1)])
    comoving_edges =  [cosmology.comoving_distance(x_) for x_ in np.array((z_bin_edges))]
    un_ = comoving_edges[0].unit
    comoving_edges = np.array([c.value for c in comoving_edges])
    comoving_edges = comoving_edges*un_

    # IA factor
    c1 = (5e-14 * (u.Mpc**3.)/(u.solMass * u.littleh**2) ) 
    c1_cgs = (c1* ((u.littleh/(cosmology.H0.value/100))**2.)).cgs
    rho_c1 = (c1_cgs*cosmology.critical_density(0)).value

    # redshift distributions
    mu = fits.open(config['2PT_FILE'])
    redshift_distributions_sources = {'z':None,'bins':dict()}
    redshift_distributions_sources['z'] = mu[6].data['Z_MID']
    for ix in config['sources_bins']:
        redshift_distributions_sources['bins'][ix] = mu[6].data['BIN{0}'.format(ix+1)]

    print ('done')    

    outputs = '/pscratch/sd/m/mgatti/Abacus/'
    if not os.path.exists(outputs+sim):
        os.mkdir(outputs+sim)
    if not os.path.exists(outputs+sim+'/intermediate/'):
        os.mkdir(outputs+sim+'/intermediate/')



    import gc
    for i in range(len(z_bin_edges)-1):
        path_ = outputs+sim+'/intermediate/g1g2_{0}.fits'.format(i)
        if not os.path.exists(path_):
            if i<10:
                m = asdf.open(path+sim+'kappa_0000{0}.asdf'.format(i))
            else:
                m = asdf.open(path+sim+'kappa_000{0}.asdf'.format(i))
            kappa_out_ = hp.ud_grade(m['data']['kappa'],nside_out=config['nside'])

            del m
            gc.collect()

            # make a full sky out of it ----
            kappa_out = copy.deepcopy(kappa_out_)
            kappa_out += rotate_map_approx(kappa_out,[ 90 ,0 , 0], flip=False,nside = config['nside'])
            kappa_out += rotate_map_approx(kappa_out,[ 180 ,0 , 0], flip=False,nside = config['nside'])
            kappa_out += rotate_map_approx(kappa_out,[ 180 ,180 , 0], flip=False,nside = config['nside'])

            # compute shear e1 and e2
            g1, g2 = gk_inv(kappa_out,kappa_out*0,config['nside'],config['nside']*2)



            fits_f = Table()
            fits_f['g1'] = g1
            fits_f['g2'] = g2
            #fits_f['g1_IA'] = g1_IA
            #fits_f['g2_IA'] = g2_IA
            fits_f.write(path_)


            
            
    sources_cat = dict()
    # load the empirical depth - number density relation from the des y3 catalog
    depth_weigth = np.load('/global/cfs/cdirs/des/mass_maps/Maps_final/depth_maps_Y3_{0}_numbdensity.npy'.format(config['nside']),allow_pickle=True).item()
    for tomo_bin in config['sources_bins']:
        sources_cat[tomo_bin] = dict()

        # load des y3 catalog
        mcal_catalog = load_obj('/global/cfs/cdirs/des/mass_maps/Maps_final/data_catalogs_weighted_{0}'.format(tomo_bin))

        pix_ = convert_to_pix_coord(mcal_catalog['ra'], mcal_catalog['dec'], nside=config['nside'])
        mask = np.in1d(np.arange(hp.nside2npix(config['nside'])),pix_)


        # generate ellipticities ***********************************
        df2 = pd.DataFrame(data = {'w':mcal_catalog['w'] ,'pix_':pix_},index = pix_)
        # draw a new random number of galaxies based on the average depth - number density des y3 relation
        nn = np.random.poisson(depth_weigth[tomo_bin])

        # the following bit draws galaxy ellipticities & weights from the empirical des y3 pixel - weight - ellipticity relation
        nn[~mask]= 0
        count = 0
        nnmaxx = max(nn)
        for count in range(nnmaxx):
            if count %2 ==0:
                df3 = df2.sample(frac=1)
                df4 = df3.drop_duplicates('pix_',keep ='first').sort_index()
            else:
                df4 = df3.drop_duplicates('pix_',keep ='last').sort_index()

            pix_valid = np.arange(len(nn))[nn>0]
            df3 = df4.loc[np.unique(pix_valid)]
            if count == 0:
                w = df3['w']
                pix = df3['pix_']
            else:
                w = np.hstack([w,df3['w']])
                pix = np.hstack([pix,df3['pix_']]) 
            nn -= 1

        del df2
        del df3
        gc.collect()
        e1,e2 = random_draw_ell_from_w(w,mcal_catalog['w'],mcal_catalog['e1'],mcal_catalog['e2'])



        del mcal_catalog
        gc.collect()


        f = 1./np.sqrt(d_tomo[tomo_bin]/np.sum(nz_kernel_sample_dict[tomo_bin]))
        f = f[pix]


        # ++++++++++++++++++++++

        n_map_sc = np.zeros(hp.nside2npix(config['nside']))

        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)


        n_map_sc[unique_pix] += np.bincount(idx_rep, weights=w/f**2)

        g1_ = g1_tomo[tomo_bin][pix]
        g2_ = g2_tomo[tomo_bin][pix]


        es1,es2 = apply_random_rotation(e1/f, e2/f)
        es1a,es2a = apply_random_rotation(e1/f, e2/f)


        x1_sc,x2_sc = addSourceEllipticity({'shear1':g1_,'shear2':g2_},{'e1':es1,'e2':es2},es_colnames=("e1","e2"))


        e1r_map = np.zeros(hp.nside2npix(config['nside']))
        e2r_map = np.zeros(hp.nside2npix(config['nside']))

        e1r_map0 = np.zeros(hp.nside2npix(config['nside']))
        e2r_map0 = np.zeros(hp.nside2npix(config['nside']))

        g1_map = np.zeros(hp.nside2npix(config['nside']))
        g2_map = np.zeros(hp.nside2npix(config['nside']))

        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)




        e1r_map[unique_pix] += np.bincount(idx_rep, weights=es1*w)
        e2r_map[unique_pix] += np.bincount(idx_rep, weights=es2*w)

        e1r_map0[unique_pix] += np.bincount(idx_rep, weights=es1a*w)
        e2r_map0[unique_pix] += np.bincount(idx_rep, weights=es2a*w)


        g1_map[unique_pix] += np.bincount(idx_rep, weights= g1_*w)
        g2_map[unique_pix] += np.bincount(idx_rep, weights= g2_*w)


        mask_sims = n_map_sc != 0.
        e1r_map[mask_sims]  = e1r_map[mask_sims]/(n_map_sc[mask_sims])
        e2r_map[mask_sims] =  e2r_map[mask_sims]/(n_map_sc[mask_sims])
        e1r_map0[mask_sims]  = e1r_map0[mask_sims]/(n_map_sc[mask_sims])
        e2r_map0[mask_sims] =  e2r_map0[mask_sims]/(n_map_sc[mask_sims])
        g1_map[mask_sims]  = g1_map[mask_sims]/(n_map_sc[mask_sims])
        g2_map[mask_sims] =  g2_map[mask_sims]/(n_map_sc[mask_sims])

        #EE,BB,_   =  g2k_sphere((g1_map+e1r_map0)*nuis['m'][tomo_bin-1], (g2_map+e2r_map0)*nuis['m'][tomo_bin-1], mask_sims, nside=config['nside2'], lmax=config['nside2']*2 ,nosh=True)
       # EEn,BBn,_ =  g2k_sphere(e1r_map*nuis['m'][tomo_bin-1], e2r_map*nuis['m'][tomo_bin-1], mask_sims, nside=config['nside2'], lmax=config['nside2']*2 ,nosh=True)
        #sources_cat[rot][tomo_bin] = {'kE':EE,'kE_noise':EEn,'mask':mask_sims}


        e1_ = ((g1_map+e1r_map0))[mask_sims]
        e2_ = ((g2_map+e2r_map0))[mask_sims]
        e1n_ = ( e1r_map)[mask_sims]
        e2n_ = ( e2r_map)[mask_sims]
        idx_ = np.arange(len(mask_sims))[mask_sims]

        kE,kB,_ = g2k_sphere(((g1_map+e1r_map0)),  ((g2_map+e2r_map0)), mask, nside=config['nside'], lmax=config['nside']*2,nosh=True)
        kEN,kBN,_ = g2k_sphere(e1r_map, e2r_map, mask, nside=config['nside'], lmax=config['nside']*2,nosh=True)

        sources_cat[tomo_bin] = {'e1':e1_,'e2':e2_,'e1n':e1n_,'e2n':e2n_,'pix':idx_,'kE':kE[mask_sims],'kEN':kEN[mask_sims]}

    #save ---
    np.save(outputs+sim+'/desy3_noSC_noIA',sources_cat)

if __name__ == '__main__':  
    from mpi4py import MPI 
    run_count = 0
    while run_count<25:
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count+comm.rank)<25:
            
            run_it(run_count+comm.rank)
            
        #        print ('failed ',runstodo[run_count+comm.rank])
             #   pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
       
#srun --nodes=4 --tasks-per-node=3 --cpus-per-task=20 --cpu-bind=cores  python make_shear.py
