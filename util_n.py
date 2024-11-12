import pandas as pd
import numpy as np

def get_SiO2_index(wavelength):
    sio2_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\SiO2_n.csv')
    # Interpolate the refractive index
    n_sio2 = np.interp(wavelength,  pd.to_numeric(sio2_data['wl']),  pd.to_numeric(sio2_data['n']))
    
    sio2_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\SiO2_k.csv')
    # Interpolate the refractive index
    k_sio2 = np.interp(wavelength,  pd.to_numeric(sio2_data['wl']),  pd.to_numeric(sio2_data['k'])) 

    return n_sio2 - 1j* k_sio2

def get_Si_index(wavelength):
    si_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\Si_n.csv')
    # Interpolate the refractive index
    n_si = np.interp(wavelength,  pd.to_numeric(si_data['wl']),  pd.to_numeric(si_data['n']))
    
    si_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\Si_k.csv')
    # Interpolate the refractive index
    k_si = np.interp(wavelength,  pd.to_numeric(si_data['wl']),  pd.to_numeric(si_data['k'])) 

    return n_si - 1j* k_si

def get_Au_index(wavelength):
    au_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\Evaporated Gold_Olmon_n.csv')
    n_au = np.interp(wavelength, au_data['wl'], au_data['n']) 
    
    au_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\Evaporated Gold_Olmon_k.csv')
    k_au = np.interp(wavelength, au_data['wl'], au_data['k']) 

    return n_au - 1j*k_au

def get_graphene_index_005eV(wavelength):
    graphene_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\Graphene\\Graphene_SJ\\GR_005ev_n.csv')
    # Interpolate the refractive index
    n_graphene = np.interp(wavelength,  pd.to_numeric(graphene_data['wl']),  pd.to_numeric(graphene_data['n']))
    
    graphene_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\Graphene\\Graphene_SJ\\GR_005ev_k.csv')
    # Interpolate the refractive index
    k_graphene = np.interp(wavelength,  pd.to_numeric(graphene_data['wl']),  pd.to_numeric(graphene_data['k'])) 

    return n_graphene - 1j* k_graphene

def get_graphene_index_060eV(wavelength):
    graphene_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\Graphene\\Graphene_SJ\\GR_060ev_n.csv')
    # Interpolate the refractive index
    n_graphene = np.interp(wavelength,  pd.to_numeric(graphene_data['wl']),  pd.to_numeric(graphene_data['n']))
    
    graphene_data = pd.read_csv('C:\\Users\\user\\DHCG\\Refractive Index\\Graphene\\Graphene_SJ\\GR_060ev_k.csv')
    # Interpolate the refractive index
    k_graphene = np.interp(wavelength,  pd.to_numeric(graphene_data['wl']),  pd.to_numeric(graphene_data['k'])) 

    return n_graphene - 1j* k_graphene

def get_Si_index():
    return 3.5