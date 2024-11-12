import torch
import meent
from util_n import *

def create_solver(fto, pol, wavelength, period_x, period_y, d, t):
    backend = 2  # Torch 0 means Jax, 1 means numpy(Fastest, can't be autograd), 2 means torch
    device = 0 # 0 : CPU, 1 : GPU
    pol = pol # 0: TE, 1: TM

    n_top = 1   # n_incidence = Air
    n_bot = 1   # get_Au_index(wavelength.item()/1000)  # n_transmission = Au
    
    theta = 0 * torch.pi / 180  # angle of incidence
    phi_rcwa = 0 * torch.pi / 180  # angle of rotation

    graphene_thickness = torch.tensor(0.34, dtype=torch.float64)
    d_Au = torch.tensor(1000, dtype=torch.float64)

    # thickness of each layer, from the first layer Air to Au

    thickness = torch.stack([t, graphene_thickness, d, d_Au])

    period = [period_x, period_y]  # length of the unit cell. Here it's 1D.

    type_complex = torch.complex128

    #Fourier order is subject to change
    mee = meent.call_mee(backend=backend, pol=pol,
                        n_top=n_top, n_bot=n_bot, theta=theta, phi=phi_rcwa,
                        fto=fto, wavelength=wavelength, period=period,
                        thickness=thickness, type_complex=type_complex,
                        device=device, fourier_type=0)

    return mee

def forward_single(mee, input_length_x,input_length_y, centers, n_indices):

    length_x = []
    length_y = []
    
    
    for length in input_length_x:
        length_x.append(length.type(torch.complex128))
    
    for length in input_length_y:
        length_y.append(length.type(torch.complex128))

    # Implement Silicon Blocks
    Si_block_1 = ['rectangle',*centers[0], length_x[0], length_y[0], n_indices[0], 0, 0, 0] # mee.rectangle(*centers[0], length_x[0], length_y[0], n_indices[0])
    Si_block_2 = ['rectangle',*centers[1], length_x[1], length_y[1], n_indices[1], 0, 0, 0] # mee.rectangle(*centers[1], length_x[1], length_y[1], n_indices[1])

    Si_blocks_list = [Si_block_1, Si_block_2]

    # Implement graphene layer
    Graphene = ['rectangle',*centers[2], length_x[2], length_y[2], n_indices[2], 0, 0, 0] # mee.rectangle(*centers[2], length_x[2], length_y[2], n_indices[2])

    # Implement graphene layer
    SiO2 = ['rectangle',*centers[3], length_x[3], length_y[3], n_indices[3], 0, 0, 0] # mee.rectangle(*centers[3], length_x[3], length_y[3], n_indices[3])

    # Implement gold back reflector
    Au = ['rectangle',*centers[4], length_x[4], length_y[4], n_indices[4], 0, 0, 0] # mee.rectangle(*centers[2], length_x[2], length_y[2], n_indices[2])

    layer_info_list = [[n_indices[-1], Si_blocks_list], [n_indices[2], [Graphene]], [n_indices[3], [SiO2]], [n_indices[4], [Au]]] # Si blocks, graphene, SiO2 in order

    mee.ucell = layer_info_list
    
    de_ri, de_ti = mee.conv_solve()

    return de_ri, de_ti