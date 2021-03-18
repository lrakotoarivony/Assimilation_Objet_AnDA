import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.patches as pat


def ellipseToGaussian(x_center,y_center,r_xi,r_eta,alpha_rad):
    mux= x_center
    muy = y_center 
    sigma_xi = (1/6)*r_xi
    sigma_eta = (1/6)*r_eta
    
    cov = 0.5*np.tan(2*alpha_rad)*((sigma_xi**2) - (sigma_eta**2))
    CovMat = np.array([[sigma_eta**2, 0],[0, sigma_xi**2]])
    
    P = np.array([[np.cos(alpha_rad+(np.pi/2)), np.sin(alpha_rad+(np.pi/2))],[-np.sin(alpha_rad+(np.pi/2)), np.cos(alpha_rad+(np.pi/2))]])
    P_1 = np.linalg.inv(P)
    CovMatBis = np.linalg.multi_dot([P_1, CovMat, P])
    mu = np.array([mux,muy])
    
    return mu, CovMatBis

def sqrtm_maison(M):
    """M is a stack of positive definite matrices of size 2x2 (...,2,2)"""
    tau = np.trace(M, axis1=-2, axis2=-1)
    delta = M[...,0,0]*M[...,1,1] - M[...,1,0]*M[...,0,1] # determinant
    s = np.sqrt(delta)
    t = np.sqrt(tau + 2*s)
    return (M + (s*np.eye(2)[...,np.newaxis]).T) / t[...,np.newaxis,np.newaxis]

def wasserstein_metric(mu1,mu2,covMat1,covMat2):
    """mus of shape Nx2, covMats of shape Nx2x2
    returns the wasserstein distance of each pair of gaussian, array of shape N"""
    rC2 = sqrtm_maison(covMat2)
    mat = covMat1 + covMat2 - (2*sqrtm_maison(rC2 @ covMat1 @ rC2))
    wasserstein = np.linalg.norm(mu1-mu2, axis=1)**2 + np.trace(mat, axis1=1, axis2=2)
    return wasserstein

def hellinger_metric(mu1,mu2,covMat1,covMat2):
    coef = (np.linalg.det(covMat1)**(1/4))*(np.linalg.det(covMat2)**(1/4))/(np.linalg.det(0.5*(covMat1+covMat2)))
    exp_ = np.linalg.multi_dot([np.transpose(mu1-mu2),0.5*(covMat1+covMat2),mu1-mu2])
    hellinger = 1-(coef*np.exp((-1/8)*exp_))
    return hellinger

def KL_div(mu1,mu2,covMat1,covMat2):
    prod = np.linalg.multi_dot([np.transpose(mu2-mu1),np.linalg.inv(covMat2),mu2-mu1])
    tr = np.trace(np.dot(np.linalg.inv(covMat2),covMat1))
    kl = 0.5*(tr + prod - 2 + np.log(np.linalg.det(covMat2)/np.linalg.det(covMat1)))
    return kl


def direction(mean,center_ref,bruit = 0.2):
    """renvoie la direction à suivre pour arriver au centre"""
    dire = center_ref-meanw
    dire = dire/np.linalg.norm(dire, axis=1)[:,np.newaxis]
    dire += np.random.randn(mean.shape[0],2) * bruit
    return dire
    
def vitesse(covMat):
    """renvoie la vitesse du tourbillon"""
    u, s, vh = np.linalg.svd(covMat)
    return s[...,0]**0.5 + s[...,1]**0.5

def converge_center(tourbillons,center,bruit = 0.2):
    """renvoie la nouvelle position du tourbillon, le tourbillon va se diriger vers center"""
    mean = tourbillons[:,:2]
    covMat = np.insert(tourbillons[:,2:], 2, tourbillons[:,3], axis=1).reshape((-1,2,2))
    dire = direction(mean,center,bruit = bruit)
    vit = vitesse(covMat)
    return np.concatenate([mean + (vit[:,np.newaxis] * dire), tourbillons[:,2:]], axis=1)

def spiral(tourbillons,bruit = 0.2,alpha = 30):
    """renvoie la nouvelle position du tourbillon, le tourbillon va faire une rotation d'alpha ou une spirale"""
    mean = tourbillons[:,:2]
    CovMat = np.insert(tourbillons[:,2:], 2, tourbillons[:,3], axis=1).reshape((-1,2,2))
    theta = np.radians(alpha)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c))).reshape((-1,2,2))
    dire = direction(mean, np.dot(R, mean.T).reshape((-1,2)),bruit=bruit)
    vit = vitesse(CovMat)
    return np.concatenate([mean + (vit[:,np.newaxis] * dire), tourbillons[:,2:]], axis=1)

def follow_gradient(tourbillons,list_mean_gauss,list_covMat_gauss,bruit = 0.2):
    mean = tourbillons[:,:2]
    covMat = np.insert(tourbillons[:,2:], 2, tourbillons[:,3], axis=1).reshape((-1,2,2))
    dirs = [1 if (i%2 == 0) else -1 for i in range(len(list_mean_gauss))]
    dire = sum_gradient_gauss2d(mean,list_mean_gauss,list_covMat_gauss, dirs)
    new_dire = dire.copy()
    new_dire[...,0] = -dire[...,1]
    new_dire[...,1] = dire[...,0]
    new_dire = new_dire/np.linalg.norm(new_dire, axis=-1)[...,np.newaxis]
    new_dire += np.random.randn(*new_dire.shape) * bruit
    vit = vitesse(covMat)

    return np.concatenate([mean + (vit[:,np.newaxis] * new_dire), tourbillons[:,2:]], axis=1)

def generer_ellipses(N):
    """génère N ellipses, renvoie un array de taille Nx5"""
    x_center = 7*np.random.random_sample(N) + 1
    y_center = 7*np.random.random_sample(N) + 1 
    r_xi = np.random.lognormal(0,1/7,N)
    r_eta = np.random.lognormal(0,1/7,N)
    alpha = np.random.rand(N)*360
    alpha_rad = np.radians(alpha)
    return np.vstack([x_center, y_center, r_xi, r_eta, alpha_rad]).T


def ellipseToGaussian(x_center,y_center,r_xi,r_eta,alpha_rad):
    """ Transforme les ellipses en gaussiennes
    Prend des arrays de taille N en argument, renvoie un array de taille Nx5"""
    N = x_center.shape[0]
    mux= x_center
    muy = y_center 
    sigma_xi = (1/6)*r_xi
    sigma_eta = (1/6)*r_eta
    
    #CovMat = np.diag([[sigma_eta**2, 0],[0, sigma_xi**2]])
    l1 = np.vstack([sigma_eta**2, np.zeros(N)]).T
    l2 = np.vstack([np.zeros(N), sigma_xi**2]).T
    CovMat = np.concatenate([l1, l2], axis=1).reshape((-1, 2, 2))
    
    P = np.array([[np.cos(alpha_rad+(np.pi/2)), np.sin(alpha_rad+(np.pi/2))],[-np.sin(alpha_rad+(np.pi/2)), np.cos(alpha_rad+(np.pi/2))]])
    P = np.moveaxis(P, (0,1,2), (1,2,0))
    P_1 = np.linalg.inv(P)
    CovMatBis = P_1 @ CovMat @ P
    
    return np.vstack([mux, muy, CovMatBis[:,0,0], CovMatBis[:,0,1], CovMatBis[:,1,1]]).T

def step(tourbillons, list_mean_gauss, list_covMat_gauss, bruit=0.2, center=None, alpha=None, model="follow gradient"):
    """Calcule le prochain état des tourbillons
    tourbillons de taille Nx5"""
    if model == "spiral":
        next_tourbillons = spiral(tourbillons, bruit=bruit, alpha=alpha)
    elif model == "centre":
        next_tourbillons = converge_center(tourbillons, bruit=bruit, center=center)
    elif model == "follow gradient":
        next_tourbillons = follow_gradient(tourbillons, list_mean_gauss, list_covMat_gauss)
    return next_tourbillons

def generer_catalogue(N, means, covMats, bruit=0.2, center=None, alpha=None, model="follow gradient"):
    """ Le catalogue est un array de taille Nx2x10 : nb de tourbillons x (anologues,successeurs) x nb de paramètres
    renvoie un catalogue de taille Nx2x10"""
    catalogue = np.empty((N,2,10))
    ellipses = generer_ellipses(N)
    gaussians = ellipseToGaussian(ellipses[:,0], ellipses[:,1], ellipses[:,2], ellipses[:,3], ellipses[:,4]) # taille Nx5
    next_gaussians = step(gaussians, means, covMats, bruit=bruit, center=center, alpha=alpha, model=model)
    next2_gaussians = step(next_gaussians, means, covMats, bruit=bruit, center=center, alpha=alpha, model=model)
    catalogue[:,0,:5] = gaussians
    catalogue[:,0,5:] = next_gaussians
    catalogue[:,1,:5] = next_gaussians
    catalogue[:,1,5:] = next2_gaussians
    return catalogue


def plot_tourbillon(tourbillon):
    # returns a matplotlib patch
    # tourbillon de taille 5
    mean = tourbillon[:2]
    C = np.array([[tourbillon[2], tourbillon[3]],[tourbillon[3],tourbillon[4]]])
    u, s, vh = np.linalg.svd(C)
    theta = np.sign(C[1,0]) * np.arccos(np.trace(C)/2)
    a = pat.Ellipse(xy=mean, width=6*s[0]**0.5, height=6*s[1]**0.5, angle=theta*180/np.pi, alpha=0.2)
    return a

def plot_trajectory(traj_tourbillon, ax, color="red"):
    """traj_tourbillon of shape Nx5, ax is a matplotlib axe"""
    for tourbillon in traj_tourbillon:
        patch = plot_tourbillon(tourbillon)
        patch.set_facecolor(color)
        ax.add_artist(patch)
    return ax


from numpy.linalg import det
from numpy.linalg import inv

def gauss2d(pos, mean, covMat, dirs):
    """pos de taille (...,2), mean de taille (N,2), covMat de taille (N,2,2), dirs array de -1 ou 1 de taille (N)
    renvoie un array de taille (...,N) qui à chaque position associe la valeur des gaussiennes"""
    pos_centered = pos[...,np.newaxis,:]-mean
    inv_covMat = np.linalg.inv(covMat)
    pos_inv_cov = np.swapaxes(np.diagonal(np.dot(pos_centered,inv_covMat), axis1=-3, axis2=-2), -2, -1)
    gaussians = dirs*np.exp(-0.5*np.sum(pos_inv_cov*pos_centered, axis=-1)) / (2.*np.pi*np.sqrt(np.linalg.det(covMat)))
    return gaussians

def gradient_gauss2d(pos, mean, covMat, dirs):
    """pos de taille (...,2), mean de taille (N,2), covMat de taille (N,2,2), dirs array de -1 ou 1 de taille (N)
    renvoie un array de taille (...,N,2) qui à chaque position associe le gradient"""
    gaussians = gauss2d(pos, mean, covMat, dirs)
    pos_centered = pos[...,np.newaxis,:]-mean
    inv_covMat = np.linalg.inv(covMat)
    deriv_gaussians = np.swapaxes(np.diagonal(np.dot(pos_centered, inv_covMat), axis1=-3,axis2=-2),-2,-1) * gaussians[...,np.newaxis]
    return deriv_gaussians
    
def sum_gaus2d(pos, mean, covMat, dirs):
    return gauss2d(pos, mean, covMat, dirs).sum(axis=-1)

def sum_gradient_gauss2d(pos, mean, covMat, dirs):
    """renvoie un array de taille (...,2)"""
    return gradient_gauss2d(pos, mean, covMat, dirs).sum(axis=-2)

def playground():
    list_mean = []
    list_covMat = []
    while len(list_mean)<6:
        mean = np.random.randint(2.5,7.5, size=2)
        line_cov = np.random.randint(-10,10, size=3)
        covMat = np.array([[line_cov[0],line_cov[1]],[line_cov[1],line_cov[2]]])
        if(det(covMat)>0 and line_cov[0]>0):
            list_mean.append(mean)
            list_covMat.append(covMat)

    precision = 100
    X = np.linspace(0, 10, precision)
    Y = np.linspace(0, 10 , precision)
    dirs = [1 if (i%2 == 0) else -1 for i in range(len(list_mean))]
    x, y = np.meshgrid(X,Y)
    pos = np.stack([x,y], axis=-1)
    Z = sum_gaus2d(pos,list_mean,list_covMat, dirs)
    return list_mean, list_covMat, X, Y, Z


def wasserstein(u,v):
    """u,v de taille Nx5"""
    mean1 = u[:,:2]
    mean2 = v[:,:2]
    cov1 = np.insert(u[:,2:], 2, u[:,3], axis=1).reshape((-1,2,2))
    cov2 = np.insert(v[:,2:], 2, v[:,3], axis=1).reshape((-1,2,2))
    return wasserstein_metric(mean1, mean2, cov1, cov2)

def new_wasserstein(u, v):
    """u,v de taille Nx10, renvoie un tableau de taille N"""
    return wasserstein(u[...,:5], v[...,:5]) + wasserstein(u[...,5:], v[...,5:])

def compute_weights(distances, l=1.):
    kernels = np.exp(-(distances/l)**2)
    return kernels / np.sum(kernels)


def locally_constant_mean(x, neighbors, successors, weights):
    return np.sum(successors * weights.reshape((-1,1)), axis=0)

def locally_incremental_mean(x, neighbors, successors, weights):
    return x + np.sum((successors - neighbors) * weights.reshape((-1,1)), axis=0)

def locally_linear_mean(x, neighbors, successors, weights):
    W = np.diag(weights)
    X = neighbors
    Y = successors
    X_mean = np.sum(X*weights.reshape((-1,1)), axis=0)
    Y_mean = np.sum(Y*weights.reshape((-1,1)), axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    cov_X = X_centered.T @ W @ X_centered
    cov_X_inv = np.linalg.pinv(cov_X)
    cov_YX = Y_centered.T @ W @ X_centered
    return Y_mean + cov_YX @ cov_X_inv @ (x - X_mean)


def predictions(catalogue, observations, method, k=100):
    """predict the next state of a list of tourbillons
    observations of shape Mx10, returns array of shape Mx10"""
    tourbillons_suivant=[]
    predecesseurs = catalogue[:,0,:]
    N = predecesseurs.shape[0]
    for tourbillon in observations:
        distances = new_wasserstein(np.stack([tourbillon]*N, axis=0), predecesseurs)
        indices_wt = np.argpartition(distances, k)[:k] # indices of the k nearest neighbors
        neighbors = catalogue[indices_wt,0,:]
        successors = catalogue[indices_wt,1,:]
        distances_neighbors = distances[indices_wt]
        weights = compute_weights(distances_neighbors, l=np.median(distances_neighbors))
        pred = method(tourbillon, neighbors, successors, weights)
        tourbillons_suivant.append(pred)
    return np.array(tourbillons_suivant)



def list_prediction(catalogue,nb_predictions, observations, method,nb_voisin = 100):
    """construit une matrice de taille nombre d'ellipses x nb_predictions x 10
    cette matrice représente les valeurs prédites"""
    mat_prediction = np.empty((observations.shape[0],nb_predictions,observations.shape[1]))
    next_obs = observations
    for j in range(nb_predictions):
        next_obs = predictions(catalogue,next_obs, method, k=nb_voisin)
        mat_prediction[:,j] = next_obs
    return mat_prediction

def list_true_value(catalogue, means, covMats, nb_predictions,observations,bruit=0.2,center=None,alpha=None,model="follow gradient"):
    """construit une matrice de taille nombre d'ellipses x nb_predictions x 5
    cette matrice représente les valeurs réelles si les tourbillons suivent le modèle prédéfini"""
    mat_true = np.empty((observations.shape[0],nb_predictions,5))
    next_gaussians = observations[:,5:]
    for j in range(nb_predictions):
        next_gaussians = step(next_gaussians, means, covMats, bruit=bruit, center=center, alpha=alpha, model=model)
        mat_true[:,j] = next_gaussians
    return mat_true

def AnDA_RMSE(a,b):
    """ Compute the Root Mean Square Error between 2 n-dimensional vectors. """
    return np.sqrt(np.mean((a-b)**2, axis=-1))