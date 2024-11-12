from util_n import *
from util_meent import *
import time

target_wl = 5500
n_Si = get_Si_index()

def find_target_wl(center, length_x, length_y, period_x, period_y, d, t, ev):

    wavelengths = torch.linspace(target_wl-500, target_wl+500, 20)

    min = 1

    for i in range(len(wavelengths)):
        wl = wavelengths[i]
        if ev == 0:
            n_index = [n_Si, n_Si, get_graphene_index_005eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1]
        else: 
            n_index = [n_Si, n_Si, get_graphene_index_060eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1]

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl, period_x=period_x, period_y=period_y, d = d, t=t)
        
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        if de_ri.sum()<min:
            min = de_ri.sum()
            min_wl = wl

    return min_wl

def find_target_wl_max(center, length_x, length_y, period_x, period_y, d, t, ev):

    wavelengths = torch.linspace(target_wl-500, target_wl+500, 20)

    max = 0

    for i in range(len(wavelengths)):
        wl = wavelengths[i]
        if ev == 0:
            n_index = [n_Si, n_Si, get_graphene_index_005eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1]
        else: 
            n_index = [n_Si, n_Si, get_graphene_index_060eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1]

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl, period_x=period_x, period_y=period_y, d = d, t=t)
        
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        if de_ri.sum()>max:
            max = de_ri.sum()
            max_wl = wl

    return max_wl

def find_max_wl_005eV(center, length_x, length_y, period_x, period_y, d, t):

    target = find_target_wl_max(center, length_x, length_y, period_x, period_y, d, t, 0)

    wl = target.clone()
    wl.requires_grad = True

    opt = torch.optim.Adam([wl], lr = 10)

    for i in range(50):

        n_index = [n_Si, n_Si, get_graphene_index_005eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1] # Imaginary part creating error

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl, period_x=period_x, period_y=period_y, d = d, t=t)
        
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        fom = -1*de_ri.sum()

        fom.backward(retain_graph=True)
        opt.step()
        opt.zero_grad()
    
    #if(wl.item()<(target_wl-500)) or (wl.item()>(target_wl+500)):
        # return None, None


    return wl, -1*fom

def find_max_wl_060eV(center, length_x, length_y, period_x, period_y, d, t):

    target = find_target_wl_max(center, length_x, length_y, period_x, period_y, d, t, 0)

    wl = target.clone()
    wl.requires_grad = True

    opt = torch.optim.Adam([wl], lr = 10)

    for i in range(50):

        n_index = [n_Si, n_Si, get_graphene_index_060eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1] # Imaginary part creating error

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl, period_x=period_x, period_y=period_y, d = d, t=t)
        
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        fom = -1*de_ri.sum()

        fom.backward(retain_graph=True)
        opt.step()
        opt.zero_grad()
    
    #if(wl.item()<(target_wl-500)) or (wl.item()>(target_wl+500)):
        # return None, None


    return wl, -1*fom

def find_min_wl_005eV(center, length_x, length_y, period_x, period_y, d, t):

    target = find_target_wl(center, length_x, length_y, period_x, period_y, d, t, 0)

    wl = target.clone()
    wl.requires_grad = True

    opt = torch.optim.Adam([wl], lr = 10)

    for i in range(50):

        n_index = [n_Si, n_Si, get_graphene_index_005eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1] # Imaginary part creating error

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl, period_x=period_x, period_y=period_y, d = d, t=t)
        
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        fom = de_ri.sum()

        fom.backward(retain_graph=True)
        opt.step()
        opt.zero_grad()
    
    #if(wl.item()<(target_wl-500)) or (wl.item()>(target_wl+500)):
        # return None, None


    return wl, fom

def find_min_wl_060eV(center, length_x, length_y, period_x, period_y, d, t):

    target = find_target_wl(center, length_x, length_y, period_x, period_y, d, t, 1)

    wl = target.clone()
    wl.requires_grad = True

    opt = torch.optim.Adam([wl], lr = 10)

    for i in range(50):

        n_index = [n_Si, n_Si, get_graphene_index_060eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1]

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl, period_x=period_x, period_y=period_y, d = d, t=t)
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        fom = de_ri.sum()

        fom.backward(retain_graph=True)
    
        opt.step()
        opt.zero_grad()
    
    return wl, fom

def find_FWHM_005eV(peak_wl, center, length_x, length_y, period_x, period_y, d, t):

    n_index = [n_Si, n_Si, get_graphene_index_005eV(peak_wl.item()/1000), get_SiO2_index(peak_wl.item()/1000), get_Au_index(peak_wl.item()/1000),1]
    solver = create_solver(fto=[10, 0], pol = 0, wavelength=peak_wl, period_x=period_x, period_y=period_y, d = d, t=t)
    de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

    _, max_ri = find_max_wl_005eV(center, length_x, length_y, period_x, period_y, d, t)

    obj = (max_ri.item() + de_ri.sum())/2

    temp = peak_wl.item()

    wl_f = torch.tensor(temp - 50, requires_grad = True, dtype = torch.float64)
    opt = torch.optim.Adam([wl_f], lr = 1)

    for i in range(50):

        n_index = [n_Si, n_Si, get_graphene_index_005eV(wl_f.item()/1000), get_SiO2_index(wl_f.item()/1000), get_Au_index(wl_f.item()/1000),1]

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl_f, period_x=period_x, period_y=period_y, d = d, t=t)
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        fom = torch.abs(de_ri.sum()-obj)
        if (fom < torch.tensor(1e-2)):
            break

        fom.backward(retain_graph=True)
    
        opt.step()
        opt.zero_grad()
    
    
    n_index = [n_Si, n_Si, get_graphene_index_005eV(wl_f.item()/1000), get_SiO2_index(wl_f.item()/1000), get_Au_index(wl_f.item()/1000), 1]
    solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl_f, period_x=period_x, period_y=period_y, d = d, t=t)
    de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)
    
    freq_f = 2*torch.pi/wl_f
  
    wl_b = torch.tensor(temp + 50, requires_grad = True, dtype = torch.float64)

    opt = torch.optim.Adam([wl_b], lr = 1)

    for i in range(50):

        n_index = [n_Si, n_Si, get_graphene_index_005eV(wl_b.item()/1000), get_SiO2_index(wl_b.item()/1000), get_Au_index(wl_b.item()/1000), 1]

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl_b, period_x=period_x, period_y=period_y, d = d, t=t)
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        fom = torch.abs(de_ri.sum()-obj)
        if (fom < torch.tensor(1e-2)):
            break

        fom.backward(retain_graph=True)
    
        opt.step()
        opt.zero_grad()
    
    freq_b = 2*torch.pi/wl_b

    n_index = [n_Si, n_Si, get_graphene_index_005eV(wl_b.item()/1000), get_SiO2_index(wl_b.item()/1000), get_Au_index(wl_b.item()/1000), 1]
    solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl_b, period_x=period_x, period_y=period_y, d = d, t=t)
    de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

    if wl_b.item() < temp or wl_f.item() > temp:
        return None
    
    return freq_f-freq_b

def find_FWHM_060eV(peak_wl, center, length_x, length_y, period_x, period_y, d, t):

    n_index = [n_Si, n_Si, get_graphene_index_060eV(peak_wl.item()/1000), get_SiO2_index(peak_wl.item()/1000), get_Au_index(peak_wl.item()/1000), 1]
    solver = create_solver(fto=[10, 0], pol = 0, wavelength=peak_wl, period_x=period_x, period_y=period_y, d = d, t=t)
    de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)
    _, max_ri = find_max_wl_005eV(center, length_x, length_y, period_x, period_y, d, t)

    obj = (max_ri.item() + de_ri.sum())/2
    temp = peak_wl.item()

    wl_f = torch.tensor(temp - 50, requires_grad = True, dtype = torch.float64)
    opt = torch.optim.Adam([wl_f], lr = 1)

    for i in range(50):

        n_index = [n_Si, n_Si, get_graphene_index_060eV(wl_f.item()/1000), get_SiO2_index(wl_f.item()/1000), get_Au_index(wl_f.item()/1000), 1]

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl_f, period_x=period_x, period_y=period_y, d = d, t=t)
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        fom = torch.abs(de_ri.sum()-obj)
        if (fom < torch.tensor(1e-2)):
            break

        fom.backward(retain_graph=True)
    
        opt.step()
        opt.zero_grad()
    
    freq_f = 2*torch.pi/wl_f

    wl_b = torch.tensor(temp + 50, requires_grad = True, dtype = torch.float64)
    opt = torch.optim.Adam([wl_b], lr = 1)

    for i in range(50):

        n_index = [n_Si, n_Si, get_graphene_index_060eV(wl_b.item()/1000), get_SiO2_index(wl_b.item()/1000), get_Au_index(wl_b.item()/1000), 1]

        solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl_b, period_x=period_x, period_y=period_y, d = d, t=t)
        de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

        fom = torch.abs(de_ri.sum()-obj)
        if (fom < torch.tensor(1e-2)):
            break

        fom.backward(retain_graph=True)
    
        opt.step()
        opt.zero_grad()
    
    freq_b = 2*torch.pi/wl_b

    if wl_b.item() < temp or wl_f.item() > temp:
        return None
    
    return freq_f-freq_b

def sim_005eV(center, length_x, length_y, period_x, period_y, d, t, wavelength):

    wl = wavelength.clone().detach()

    n_index = [n_Si, n_Si, get_graphene_index_005eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1]

    solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl, period_x=period_x, period_y=period_y, d = d, t=t)
    
    de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

    fom = de_ri.sum()
    
    return fom

def sim_060eV(center, length_x, length_y, period_x, period_y, d, t, wavelength):

    wl = wavelength.clone().detach()

    n_index = [n_Si, n_Si, get_graphene_index_060eV(wl.item()/1000), get_SiO2_index(wl.item()/1000), get_Au_index(wl.item()/1000), 1]

    solver = create_solver(fto=[10, 0], pol = 0, wavelength=wl, period_x=period_x, period_y=period_y, d = d, t=t)
    
    de_ri, _ = forward_single(solver, length_x, length_y, center, n_index)

    fom = de_ri.sum()
    
    return fom

def find_BIC():

    for counter in range(100):

        print("Finding Candidate", end = '\r')
        
        p = torch.tensor(np.random.randint(2500,3000), requires_grad=True, dtype=torch.float64)
        delta = torch.tensor(np.random.randint(1,100)/1000, requires_grad=True, dtype=torch.float64)
        d = torch.tensor(np.random.randint(1600,2000), requires_grad=True, dtype=torch.float64)
        t = torch.tensor(np.random.randint(100,300), requires_grad=True, dtype=torch.float64)
        w = torch.tensor(np.random.randint(p.item()//2.5, p.item()//1.8), requires_grad=True, dtype=torch.float64)

        period_x = 2* p
        period_y = torch.tensor(1, requires_grad=False, dtype=torch.float64)

        center = [[-1*p/2*(1-delta), period_y/2], [p/2*(1-delta), period_y/2], [period_x/2, period_y/2], [period_x/2, period_y/2], [period_x/2, period_y/2]]
        length_x = [ w, w, period_x, period_x, period_x]
        length_y = [ period_y, period_y, period_y, period_y, period_y]

        peak_005eV, min_005eV = find_min_wl_005eV(center, length_x, length_y, period_x, period_y, d, t)
        peak_060eV, min_060eV = find_min_wl_060eV(center, length_x, length_y, period_x, period_y, d, t)

        if peak_005eV == None or peak_060eV == None:
            continue

        center = [[-1*p/2, period_y/2], [p/2, period_y/2], [period_x/2, period_y/2], [period_x/2, period_y/2], [period_x/2, period_y/2]]

        min_005eV_dlta0 = sim_005eV(center, length_x, length_y, period_x, period_y, d, t, peak_005eV)
        min_060eV_dlta0 = sim_060eV(center, length_x, length_y, period_x, period_y, d, t, peak_060eV)

        fom = (min_005eV_dlta0+min_060eV_dlta0) - (min_005eV+min_060eV)

        if (fom.item() > 0.05):
            print("Found Candidate, fom: ", fom.item(), end = '\r')
            break
    
    if counter == 99:
        print("Couldn't find Candidnate Structure")
        return None, None, None, None, None, None

    opt = torch.optim.Adam([
        {'params': [p, w], 'lr': 20},
        {'params': [d,t], 'lr': 10},
        {'params': [delta], 'lr': 0.001}
    ])

    for counter_1 in range(100):
        
        period_x = 2 * p
        period_y = torch.tensor(1, requires_grad=False, dtype=torch.float64)

        center = [[-1*p/2*(1-delta), period_y/2], [p/2*(1-delta), period_y/2], [period_x/2, period_y/2], [period_x/2, period_y/2], [period_x/2, period_y/2]]
        length_x = [ w, w, period_x, period_x, period_x]
        length_y = [ period_y, period_y, period_y, period_y, period_y]

        peak_005eV, min_005eV = find_min_wl_005eV(center, length_x, length_y, period_x, period_y, d, t)
        peak_060eV, min_060eV = find_min_wl_060eV(center, length_x, length_y, period_x, period_y, d, t)

        center = [[-1*p/2, period_y/2], [p/2, period_y/2], [period_x/2, period_y/2], [period_x/2, period_y/2], [period_x/2, period_y/2]]

        min_005eV_dlta0 = sim_005eV(center, length_x, length_y, period_x, period_y, d, t, peak_005eV)
        min_060eV_dlta0 = sim_060eV(center, length_x, length_y, period_x, period_y, d, t, peak_060eV)

        fom = (min_005eV_dlta0+min_060eV_dlta0) - (min_005eV+min_060eV)
        (-1 * fom).backward(retain_graph=True)

        print(f"epoch: {counter_1}, FOM: {fom.item()}", end='\r')

        if (fom.item() > 0.2):
            break

        opt.step()
        opt.zero_grad()
    
    if counter_1 == 99:
        print("Couldn't find Candidnate Structure")
        return None, None, None, None, None, None
    
    return p, delta, d, t, w, fom

def find_tun(parameters):

    # Initialize parameters as tensors with requires_grad=True
    p = torch.tensor(parameters[0], requires_grad=True, dtype=torch.float64)
    delta = torch.tensor(parameters[1], requires_grad=True, dtype=torch.float64)
    d = torch.tensor(parameters[2], requires_grad=True, dtype=torch.float64)
    t = torch.tensor(parameters[3], requires_grad=True, dtype=torch.float64)
    w = torch.tensor(parameters[4], requires_grad=True, dtype=torch.float64)

    # Use a smaller learning rate for stability
    opt = torch.optim.Adam([
        {'params': [p, w], 'lr': 0.5},
        {'params': [d, t], 'lr': 0.5},
        {'params': [delta], 'lr': 0.0001}
    ])

    previous_tunability = None

    for i in range(100):

        period_x = 2 * p
        period_y = torch.tensor(1, requires_grad=False, dtype=torch.float64)

        center = [[-1 * p / 2 * (1 - delta), period_y / 2], [p / 2 * (1 - delta), period_y / 2], [period_x / 2, period_y / 2], [period_x / 2, period_y / 2], [period_x / 2, period_y / 2]]
        length_x = [w, w, period_x, period_x, period_x]
        length_y = [period_y, period_y, period_y, period_y, period_y]

        # Calculate peaks and frequencies
        peak_005eV, _ = find_min_wl_005eV(center, length_x, length_y, period_x, period_y, d, t)
        peak_060eV, _ = find_min_wl_060eV(center, length_x, length_y, period_x, period_y, d, t)

        #print(peak_005eV, peak_060eV)

        peak_freq_005eV = 2 * torch.pi / peak_005eV
        peak_freq_060eV = 2 * torch.pi / peak_060eV

        FWHM_060 = find_FWHM_060eV(peak_060eV, center, length_x, length_y, period_x, period_y, d, t)
        FWHM_005 = find_FWHM_005eV(peak_005eV, center, length_x, length_y, period_x, period_y, d, t)

        if (FWHM_005 != None) and (FWHM_060 != None):
            tunability = (-peak_freq_060eV + peak_freq_005eV) * 2 / (FWHM_060 + FWHM_005 + 1e-10)  # Add small constant to avoid division issues
        else: 
            print("Ineffective FWHM, restart with new trial")
            return None, None, None, None, None, None

        tunability.backward()

        print(f"Iteration {i}, FOM: {tunability.item()}, "
              f"Parameters - p: {p.item()}, delta: {delta.item()}, d: {d.item()}, t: {t.item()}, w: {w.item()}", 
              f"Gradients - p: {p.grad}, delta: {delta.grad}, d: {d.grad}, t: {t.grad}, w: {w.grad}", 
              end='\r')

        # Check for stopping condition based on improvement threshold
        if previous_tunability is not None and abs(tunability.item() - previous_tunability) < 5e-3:
            break
        previous_tunability = tunability.item()

        opt.step()
        opt.zero_grad()
        time.sleep(0.5)

    return p, delta, d, t, w, -1 * tunability.item()

def tunability(parameters):

    # Initialize parameters as tensors with requires_grad=True
    p = torch.tensor(parameters[0], requires_grad=True, dtype=torch.float64)
    delta = torch.tensor(parameters[1], requires_grad=True, dtype=torch.float64)
    d = torch.tensor(parameters[2], requires_grad=True, dtype=torch.float64)
    t = torch.tensor(parameters[3], requires_grad=True, dtype=torch.float64)
    w = torch.tensor(parameters[4], requires_grad=True, dtype=torch.float64)

    period_x = 2 * p
    period_y = torch.tensor(1, requires_grad=False, dtype=torch.float64)

    center = [[-1 * p / 2 * (1 - delta), period_y / 2], [p / 2 * (1 - delta), period_y / 2], [period_x / 2, period_y / 2], [period_x / 2, period_y / 2], [period_x / 2, period_y / 2]]
    length_x = [w, w, period_x, period_x, period_x]
    length_y = [period_y, period_y, period_y, period_y, period_y]

    # Calculate peaks and frequencies
    peak_005eV, _ = find_min_wl_005eV(center, length_x, length_y, period_x, period_y, d, t)
    peak_060eV, _ = find_min_wl_060eV(center, length_x, length_y, period_x, period_y, d, t)

    #print(peak_005eV, peak_060eV)

    peak_freq_005eV = 2 * torch.pi / peak_005eV
    peak_freq_060eV = 2 * torch.pi / peak_060eV
    
    FWHM_060 = find_FWHM_060eV(peak_060eV, center, length_x, length_y, period_x, period_y, d, t)
    FWHM_005 = find_FWHM_005eV(peak_005eV, center, length_x, length_y, period_x, period_y, d, t)
    tunability = (-peak_freq_060eV + peak_freq_005eV) * 2 / (FWHM_060 + FWHM_005 + 1e-10)  # Add small constant to avoid division issues

    if (FWHM_005 != None) and (FWHM_060 != None):
        tunability = (-peak_freq_060eV + peak_freq_005eV) * 2 / (FWHM_060 + FWHM_005 + 1e-10)  # Add small constant to avoid division issues
    else: 
        return 0
    
    return -1 * tunability.item()
