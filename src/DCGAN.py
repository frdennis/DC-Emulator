import torch
import numpy as np
import readgadget
from tqdm import tqdm
import scipy.integrate as integrate

import os
import shutil 
import subprocess

from .model import AUNetG
from .utils import *


class DCGAN():
    def __init__(self, state_path="src/state_142.pt",
            device="cpu"):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        state = torch.load(state_path, map_location=torch.device('cpu'))

        self.model = AUNetG(in_chan=3, out_chan=3, style_size=5, scale_factor=2).to(device)
        self.model.load_state_dict(state['model'])
        self.model.eval()

        self.noise_arr = []
        noise_sdims = [142, 142, 140, 72, 72, 70, 37, 37, 
                        37, 37, 37, 72, 72, 72, 72, 142, 142, 142]
        for n in noise_sdims:
            self.noise_arr.append(torch.randn(1,1,n,n,n))

    
    def upscale(self, x, s=None, n=None,
            a=None, z=None,
            Om_k=None, sigma8=None, Om_m=None, h=None):
        """Upscale input field x for a given style s with optional noise array n

        Parameters:
            - x (numpy array [3, N, N, N])   :   input displacement field
            - s (numpy array [5])            :   style parameters
            - n (list)                          :   specific noise realization

        Returns:
            - y (numpy array [3, 2N, 2N, 2N])    :   predicted upscaled displacement field
        """
        # Convert redshift to scale factor if given
        if z is not None and a is None:
            a = 1.0 / (1.0 + z)

        # Handle style
        if s is not None:
            s = np.asarray(s)

            if s.size != 5:
                raise ValueError(
                    "Style input s must contain exactly 5 parameters: "
                    "[a, Om_k, sigma8, Om_m, h]"
                )

            s = torch.tensor(
                s, dtype=torch.float32,
                device=self.device
            )#.reshape(1,5)
        else:
            required = {
                "a": a,
                "Om_k": Om_k, 
                "sigma8": sigma8,
                "Om_m": Om_m,
                "h": h, 
            }

            missing = [key for key, value in required.items() if value is None]

            if len(missing) > 0:
                raise ValueError(
                    f"Missing parameter(s): {', '.join(missing)}. "
                )

            s = torch.tensor(
                [a, Om_k, sigma8, Om_m, h],
                dtype=torch.float32,
                device=self.device
            )

        #if s is not None and len(style) == 5:
        #    s = torch.from_numpy(s).to(self.device).to(torch.float32)   
        #style = torch.tensor([a, Om_k, sigma8, Om_m, h], dtype=torch.float32)

        x = torch.from_numpy(x).to(self.device).to(torch.float32)
        #s = torch.from_numpy(s).to(self.device).to(torch.float32)

        if n == True:           
            n = self.noise_arr  # Use noise array from init
        elif n == None:
            n = [None,]*18      # Tell each Addnoise to generate on the fly

        dis(x, a=s[0])
        y = torch.from_numpy(chunk_upscale(x, s, self.model, noise_arr=n))
        dis(y, a=s[0], undo=True)

        return np.array(y)


    def velocity(self, snapshot, snapshot_ic, 
        sigma8,
        dz=None, da=None, upscale=True
        ):
        header = readgadget.header(snapshot)
        
        a       = header.time
        z       = header.redshift
        Om_m    = header.omega_m
        Om_l    = header.omega_l
        h       = header.hubble

        Om_k    = 1. - (Om_m + Om_l)

        if da is not None:  # Mainly use this one
            zm = 1 / (a + da) - 1
            zp = 1 / (a - da) - 1

            dzp = zp - z
            dzm = zm - z
            dap = da 
            dam = -da
        elif dz is not None:
            ap = 1 / (1 + z + dz)
            am = 1 / (1 + z - dz)

            dzp = dz
            dzm = -dz
            dap = ap - a 
            dam = am - a
        else:
            raise ValueError("Either dz or da must be provided") 

        style = torch.tensor([a, Om_k, sigma8, Om_m, h], dtype=torch.float32)

        style_p = style.clone()
        style_p[0] = a + dap    # t+dt
        input_p = self.make_lag_dz(snapshot, snapshot_ic, dzp)

        style_m = style.clone() 
        style_m[0] = a + dam    # t-dt
        input_m = self.make_lag_dz(snapshot, snapshot_ic, dzm)

        #dt = self.t(Om_k, Om_m, Om_l, h, a + dap, a + dam) 
        dt = self.t(Om_k, Om_m, Om_l, h, a + dam, a + dap) 
        if not upscale:
            dx = input_p - input_m
            #return input_p, input_m

            velocity = dx / dt * a / h
        else:
            #input_p = torch.from_numpy(input_p).to(torch.float32)
            #input_m = torch.from_numpy(input_m).to(torch.float32)

            output_p = self.upscale(input_p, style_p, n=self.noise_arr)
            output_m = self.upscale(input_m, style_m, n=self.noise_arr)

            dx = output_p - output_m
            velocity = dx / dt * a / h

        return velocity

    def position(self, snapshot, snapshot_ic, 
            sigma8, upscale=True, noise=None):
        
        header = readgadget.header(snapshot)
        a       = header.time
        z       = header.redshift
        Om_m    = header.omega_m
        Om_l    = header.omega_l
        h       = header.hubble
        
        Om_k = 1. - (Om_m + Om_l)

        style = torch.tensor([a, Om_k, sigma8, Om_m, h], dtype=torch.float32)
        input = self.make_lag_dz(snapshot, snapshot_ic, dz=0.0)

        if not upscale: 
            return input
        else: 
            #input = torch.from_numpy(input).to(torch.float32)
            output = self.upscale(input, style, n=noise)

            #dis(input, a=style[0])
            #output = torch.from_numpy(chunk_upscale(input, style, self.model, noise_arr=self.noise_arr))
            #dis(output, undo=True, a=style[0])

            return output

    def readsnap(self, file):
        ptype = [1]

        pos = readgadget.read_block(file, "POS ", ptype)/1e3        # positions in Mpc/h
        vel = readgadget.read_block(file, "VEL ", ptype)            # peculiar velocities in km/s (sqrt(a) * u)
        ids = readgadget.read_block(file, "ID  ", ptype)  

        pos = np.array(pos)
        vel = np.array(vel)
        ids = np.array(ids)

        ids -= ids.min()                                            #IDs starting from 0

        header = readgadget.header(file)

        return pos, vel, ids, header

    def order(self, arr, ids):
        _arr = np.zeros((arr.shape[0]*8, 3)) * np.nan   # *8 since IC IDs are sourced from HR ICs with 8x more particles
        _arr[ids] = arr
        _arr = _arr[~np.isnan(_arr).any(axis=1)] # pop elements with NaN values
        return _arr

    def make_lag_dz(self, file, file_ic, dz):
        pos, vel, ids, header = self.readsnap(file)
        pos_ic, vel_ic, ids_ic, header_ic = self.readsnap(file_ic)

        Npart = pos.shape[0]
        Ngrid = int(np.rint(Npart**(1/3)))
        BoxSize = header.boxsize/1e3

        a   = header.time
        Om0 = header.omega_m #header.Omega0
        Ol  = header.omega_l #header.omegaLambda
        #h   = header.hubble
        Omk = 1. - Om0 - Ol

        """
        dx_u = -10.0 * dz * a / np.sqrt(
            ((Om0 / a) / a) + 
            (Ol * a) + 
            #((1. - (Om0 + Ol)) / (a * a * a))  # radiation
            (Omk / a)                           # curvature
        )
        """
        dx_u = -10.0 * dz * np.sqrt(a) / np.sqrt(   # need sqrt(a) here since readgadget multiplies a sqrt(a) in read_block
            ((Om0 / a) / a) + 
            (Ol * a) + 
            #((1. - (Om0 + Ol)) / (a * a * a))  # radiation
            (Omk / a)                           # curvature
        )
        
        assert Npart == header_ic.nall[1]

        pos    = self.order(pos, ids)
        vel    = self.order(vel, ids)
        q      = self.order(pos_ic, ids_ic)

        q = (q + BoxSize) % BoxSize # 
        indxs = np.rint(q / BoxSize * Ngrid) % Ngrid
        q = (indxs) / Ngrid * BoxSize

        dis = np.ones([Ngrid, Ngrid, Ngrid, 3]) * np.nan

        """
        for ii, inx in tqdm(enumerate(indxs.astype(int)), total=len(indxs)):
            i, j, k = inx

            dis[i,j,k] = (pos[ii] + vel[ii] * dx_u/1e3) - q[ii]
        """

        indxs = indxs.astype(int)

        i = indxs[:, 0]
        j = indxs[:, 1]
        k = indxs[:, 2]

        disp = (pos + vel * dx_u / 1e3) - q
        dis[i,j,k] = disp
        dis -= np.rint(dis / BoxSize) * BoxSize

        assert not np.isnan(dis).any()

        dis = np.moveaxis(dis, -1, 0)
        return dis


    def H(self, a, Om_k=0.013, Om_m=0.307, Om_l=0.7, h=0.73):

        #Om_r = 1 - (Om_l + Om_k + Om_m)
        Om_r = 0.0
        return 100*h * np.sqrt(Om_r*a**(-4) + Om_m*a**(-3) + Om_k*a**(-2) + Om_l) # km/s/Mpc

    def t(self, Om_k, Om_m, Om_l, h, a1, a2):
        #a       = style[0]
        #Om_k    = style[1]
        #Om_m    = style[3]
        #h       = style[4] #* 3.241e-20 # km/s/Mpc

        return integrate.quad(
            lambda x: 1/(x*self.H(x, Om_k, Om_m, Om_l, h)), 
            a1, a2
        )[0]
