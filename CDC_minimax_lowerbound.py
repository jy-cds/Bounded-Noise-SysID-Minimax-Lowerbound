import numpy as np
import cvxpy as cp
import tqdm
import os
import datetime as dt
import pickle
from scipy.spatial.distance import pdist


np.random.seed(123)


def sample_sphere(d,N=100):
    # dimension d and makes 10 samples
    x = np.random.normal(size=(N, d))
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    return x



def stabilize(A, max_eig=0.8):
    eigs,_ = np.linalg.eig(A)
    spec_rad_A = max(np.abs(eigs))
    A = A / spec_rad_A * max_eig
    return A



def generate_trajectory(true_theta, specs, savedir='out_unspecified'):
    # assume that control action will be generated from truncated Gaussian, if B matrix is part of the dynamics
    specs['true_theta'] = true_theta
    distribution = specs['distribution']
    T = specs['T']
    W = specs['W']
    Nx = specs['Nx']
    seed = specs['seed']
    np.random.seed(seed)
    A = true_theta['A']
    H = specs['H']
    Nw = H.shape[1]

    if distribution == "uniform":
        w = np.random.uniform(low = -W, high = W, size=(T,Nw))
    else:
        print('Distribution not supported.')
        
    x = [None]*(T+1)
    x[0] = np.random.normal(size=(Nx,))
    for t in range(T):
        x[t+1] = A @ x[t] + H @ w[t]
    trajectory = dict(A=A,  x=np.array(x), w_list=w, W=W, specs=specs)
    
    # save data   
    if savedir is not None:
        if savedir != '':
            os.makedirs(savedir, exist_ok=True)
        tz = dt.timezone(dt.timedelta(hours=-8))  # PST
        start_time = dt.datetime.now(tz)
        filename = os.path.join(savedir, f'trajectory_{distribution}_')
        filename += f'Nx={Nx}_'
        filename += f'T={T}_'
        filename += f'W={W}_'
        filename += f'seed={seed}_'
        filename += start_time.strftime('%Y%m%d_%H%M%S')
        # with open(f'{filename}.pkl', 'wb') as f:
        #     pickle.dump(file=f, obj=trajectory)
    return trajectory



def support(Z, d, G, h_lb, h_ub , lb, ub, prev_bound = None):
    y = cp.Variable((d,1))
    constraints = [y <= ub*np.ones((d,1))]
    constraints += [y >= lb*np.ones((d,1))]
    constraints += [G@y <= h_ub]
    constraints += [G@y >= h_lb]
    if prev_bound is not None:
        lower = prev_bound['lower']
        prev_Z = prev_bound['Z']
        constraints +=[prev_Z @ y <= lower.reshape((prev_Z.shape[0],1))]
        
    y_values = np.zeros(shape=(Z.shape[0], d))
    lowerbound = np.zeros((Z.shape[0],))
    for i, zi in enumerate(Z):
        prob = cp.Problem(cp.Maximize(y.T @ zi), constraints)
        prob.solve(cp.MOSEK)
        y_values[i] = y.value.flatten()
        lowerbound[i] = prob.value
    
    return y_values, lowerbound



def get_params_for_diameter(x, nx, W):
    G = np.vstack([np.kron(np.eye(nx), xt) for xt in x[:-1]])
    h_ub = np.vstack([x_.reshape((nx,1))  for x_ in x[1:]]) + W
    h_lb = np.vstack([x_.reshape((nx,1))  for x_ in x[1:]]) - W
    return G,h_lb, h_ub



def get_diam(x,u,nx,W,lb,ub, prev_bound=None, Z=None):
    '''compute the diameter of points of box surrounding points'''
    # theta = [A11,A12,A21,A22,B1,B2]
    # https://arxiv.org/pdf/1905.11877.pdf Algorithm 3.
    # d: dimension of the parameter space
    # r: Radius of a ball that contains the parameter polytope
    # epsilon: Steiner point approximation accuracy
    # x_list,u_list: state and control trajectory
    # W,lb,ub: disturbance bound, lower and upper bound on the parameter space.
    G, h_lb, h_ub = get_params_for_diameter(x,nx, W)
    d = nx*(nx)
    # mat = np.eye(nx)
    if Z is None:
        n_samp = 18*d
        Z = sample_sphere(d,N=n_samp) # Z is a N by d matrix with N iid sampled R^d gaussian vector
    # print(Z.shape)
    # Z = np.vstack([np.eye(d),-np.eye(d)])
    if prev_bound is not None:
        P, lowerbound = support(Z,d,G, h_lb, h_ub, lb, ub, prev_bound)
    else:
        P, lowerbound = support(Z,d,G, h_lb, h_ub, lb, ub)
        
    pairs = pdist(P,metric="euclidean")
    return max(pairs), lowerbound, Z



def get_SM_diam_no_skip(x, w, specs, savedir='out_unspecified', max_diam=100, w_hat = None, skip_to_end = False, optional_name = None):
    # Diameter using box approximation of the membership set
    # x: x[0]...x[T]
    # savedir: the directory to which the data will be save
    # max_diam: some rough initialization bound on the diameter for the parameters before the SM set becomes compact with enough data
    
    T = specs['T']
    W = specs['W']


    z = x
    Nx = x.shape[1]
    d = z.shape[1] * Nx # (Nx+Nu) * Nx parameters

    # initialization
    diam_SM = np.zeros((T+1,))
    diam_SM[d-1] = max_diam
    mu_t_list = [None] * (T+1)
    mu_t_list[d-1] = np.ones(shape=(2*d,)) * max_diam
    
    if skip_to_end:
        if w_hat is not None:
            W = w_hat[-1]
        diam_SM, _ , _ = get_diam(x, None, Nx, W, -100, 100)
        print('----- Seed = {}, Nx = {}, SM diam = {}, W = {}-----'.format(specs['seed'], Nx, diam_SM/np.sqrt(Nx), W))
    else:
        for t in tqdm.tqdm(range(d, T+1)): 
            if w_hat is not None:
                W = w_hat[t]
            diam, Y,_ = get_diam(x[:t], None, Nx, W, -100, 100)
            diam_SM[t] = diam


    # save data   
    if savedir is not None:
        if savedir != '':
            os.makedirs(savedir, exist_ok=True)
        distribution = specs['distribution']
        tz = dt.timezone(dt.timedelta(hours=-8))  # PST
        start_time = dt.datetime.now(tz)
        

        if w_hat is not None:
            if optional_name is not None:
                filename = os.path.join(savedir, f'SM-UCB_{optional_name}_diameter_{distribution}_')
            else:
                filename = os.path.join(savedir, f'SM-UCB_diameter_{distribution}_')
        else:
            filename = os.path.join(savedir, f'SM_diameter_{distribution}_')
        filename += f'Nx={Nx}_'
        filename += f'T={T}_'
        filename += f'W={W}_'
        filename += f'seed={seed}_'
        filename += start_time.strftime('%Y%m%d_%H%M%S')
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(file=f, obj=dict(diam_SM=diam_SM, x=np.array(x), w_list=w, W=W, specs=specs, w_hat=w_hat))
        
    return diam_SM



def get_proj_diam(x, specs):
    T = specs['T']
    W = specs['W']
    
    # get z[0],...z[T-1]
    z = x[:T]
    Nx = x.shape[1]
    Nz = z.shape[1]
    
    theta_LS = [None] * (T+1)
    for t in range(Nz,T+1):
        Gamma_t = z[:t].T @ z[:t] # Gamma_t = zt@zt' summing up to t-1
        if np.linalg.matrix_rank(Gamma_t) < Nz:
            print('Data not exciting enough for full rank!')
        theta_LS[t] = x[1:t+1].T @ z[:t] @ np.linalg.pinv(Gamma_t) 
        
        G,h_lb, h_ub = get_params_for_diameter(x[:t+1], Nx, W)
        if (G@theta_LS[t].reshape((Nx**2,1)) - 1e-6 > h_ub).any() or (G@theta_LS[t].reshape((Nx**2,1)) + 1e-6 < h_lb).any():
        
            th = cp.Variable((Nx,Nx))
            constraints = [G @ cp.reshape(th, (Nx**2,1), order='C') <= h_ub ]
            constraints += [ G @ cp.reshape(th, (Nx**2,1), order='C') >= h_lb]
            constraints += [th <= 1e5]
            constraints += [th >= -1e5]
            
            obj = cp.Minimize(cp.norm(th - theta_LS[t],p=2)**2) 
            prob = cp.Problem(obj, constraints)
            prob.solve(cp.MOSEK)
            theta_LS[t] = th.value
        
    return theta_LS



def get_LS_diam(x, specs):
    T = specs['T']
    
    # get z[0],...z[T-1]
    z = x[:T]
    Nz = z.shape[1]
    
    theta_LS = [None] * (T+1)
    for t in range(Nz,T+1):
        Gamma_t = z[:t].T @ z[:t] # Gamma_t = zt@zt' summing up to t-1
        if np.linalg.matrix_rank(Gamma_t) < Nz:
            print('Data not exciting enough for full rank!')
        theta_LS[t] = x[1:t+1].T @ z[:t] @ np.linalg.pinv(Gamma_t) 
    
    return theta_LS


  
def get_CLS_diam(x, specs, w_max=2):
    # w_max is the upper bound on the disturbances used to compute the SME uncertainty

    T = specs['T']
    
    # get z[0],...z[T-1]
    z = x[:T]
    Nx = x.shape[1]
    Nz = z.shape[1]
    
    theta_LS = [None] * (T+1)
    for t in range(Nz,T+1):
            G,h_lb, h_ub = get_params_for_diameter(x[:t+1], Nx, w_max)
            th = cp.Variable((Nx,Nx))
            constraints = [G @ cp.reshape(th, (Nx**2,1), order='C') <= h_ub ]
            constraints += [ G @ cp.reshape(th, (Nx**2,1), order='C') >= h_lb]
            constraints += [th <= 1e5]
            constraints += [th >= -1e5]
            
            xt = np.hstack([x[:t]]).T
            xt1 = np.hstack([x[1:t+1]]).T
            obj = cp.Minimize(cp.norm(xt1 - th @ xt,p='fro')**2) # unregularized LSE
            prob = cp.Problem(obj, constraints)
            prob.solve(cp.MOSEK)
            theta_LS[t] = th.value

    return theta_LS



def get_lowerbound(N, delta = 0.99, mu = 0.1):
    C_w = 1/4
    Ew = 1
    return  (1-2*delta/N)*(1-mu)/(2*C_w*Ew)



if __name__ == '__main__':
    savedir = 'CDC25'
    diam = 50
    W       = 2.
    distribution = 'uniform'
    
    
    
    rho = 0.7
    for seed in [2,3,5,7,2025]:
        np.random.seed(seed)
        
        for T in [1500,]: 
            N = 4
            A = stabilize( np.random.uniform(low = -5., high = 5.,size=(N,N)) , rho)
            H = np.eye(N)
            Nx = A.shape[1]
            Nz = Nx
            true_theta = dict(A=A)
            run_specs = dict(T=T, W=W, Nx=N, seed=seed, distribution=distribution, H=H)
            trajectory = generate_trajectory(true_theta=true_theta, specs=run_specs, savedir=None)
   

        
            # get OLS-SME (projection) error
            PROJ_theta = get_proj_diam(x=trajectory['x'], specs=trajectory['specs'])
            PROJ_error = [None]*N
            for theta_blend in PROJ_theta[N:]:
                PROJ_error.append( np.linalg.norm(theta_blend - A) )
                
            # get OLS error    
            OLS_theta = get_LS_diam(x=trajectory['x'],  specs=trajectory['specs'])
            OLS_error= [None]*N
            for theta in OLS_theta[N:]:
                OLS_error.append( np.linalg.norm(theta - A) )             
            
            
            # get CLS error
            CLS_theta = get_CLS_diam(x=trajectory['x'], specs=trajectory['specs'], w_max = W)
            CLS_error = [None]*N
            for CLS_loose in CLS_theta[N:]:
                CLS_error.append( np.linalg.norm(CLS_loose - A) )
                
        
            # get SME uncertainty set diameter (worst case error bound for SME-based estimators)
            diam = get_SM_diam_no_skip(x=trajectory['x'],  w=trajectory['w_list'], specs=trajectory['specs'], savedir=None, skip_to_end = False)
                
            # get theoretical bound
            lb_multiplier = get_lowerbound(N, delta = 0.99, mu = 0.1)
            LB_error = lb_multiplier * np.array([1 / i for i in range(N, T + 1)])


            if savedir != '':
                os.makedirs(savedir, exist_ok=True)

            filename = os.path.join(savedir, f'{distribution}_')
            filename += f'Nx={Nx}_'
            filename += f'T={T}_'
            filename += f'W={W}_'
            filename += f'seed={seed}_'
            filename += f'rho={rho}'
            if savedir is not None:
                with open(f'{filename}.pkl', 'wb') as f:
                    pickle.dump(file=f, obj=dict(PROJ=PROJ_error, OLS=OLS_error, CLS=CLS_error, SME=diam, LB=LB_error,))            
            