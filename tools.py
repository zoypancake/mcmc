import numpy as np
from scipy.stats import norm

def PDFt4(x, m, s):
    z = (x-m)/s
    pdf = 3/(4*1.414213562*s)*pow(1 + z*z/2, -2.5)
    return(pdf)

def random_bac():
    mBactrian = 0.95; sBactrian = np.sqrt(1-mBactrian**2)
    z = mBactrian + np.random.normal(loc=0.0, scale=1.0, size=1)*sBactrian
    if np.random.random(1) < 0.5 :
        z = -z
    return(z)

def rndTriangle():
    #Standard Triangle variate, generated using inverse CDF
    u = np.random.random(1);
    if(u > 0.5):
        z =  np.sqrt(6.0) - 2.0*np.sqrt(3.0*(1.0 - u));
    else:
        z = -np.sqrt(6.0) + 2.0*np.sqrt(3.0*u);
    return z;

def random_bac_tri():
    mBactrian = 0.95; sBactrian = np.sqrt(1-mBactrian**2)
    z = mBactrian + rndTriangle()*sBactrian;
    if(np.random.random(1) < 0.5):
        z = -z;
    return (z)

def sample_Disc2D(size, radius, dimension):
    direction = np.random.normal(size=(dimension,size))
    direction /= np.linalg.norm(direction, axis=0)
    rad = pow(np.random.random(size),(1/dimension))
    return radius * (rad * direction).T


def Eff_IntegratedCorrelationTime (Y):
#     This calculates Efficiency or Tint using Geyer's (1992) initial positive
#     sequence method.
#     Note that this destroys x[].
    Tint=1; rho0=0; n = len(Y); x = Y.copy()
    x -= np.mean(x)
    x /= np.std(x)
    for irho in range(1,(n-10)):
        rho = np.sum(np.multiply(x[0:n-irho], x[irho:n]))
        rho /= (n-1);
        if ((irho>10) & (rho+rho0<0)):
            break;
        Tint += rho*2;
        rho0 = rho;
    return (1/Tint);


def norm_pdf(x, loc, scale):
    p = np.exp(-(loc-x)*(loc-x)/(2*scale*scale))/(np.sqrt(2*np.pi)*scale)
    return (p)


def logtargetpdf(x, target):
    lnp = 0.0
    if (target == 'N01'):
        lnp = -x*x/2
    elif (target == 'TwoNormal'):
        lnp = np.log(0.25 * norm_pdf(x, loc=-1, scale=(1/2)) + 0.75 * norm_pdf(x, loc=1, scale=(1/2)))
    elif (target == 'TwoT4'):
        m = 0.75
        st = np.sqrt(37.0)/8.0
        lnp = np.log(0.75 * PDFt4(x,-m,st) + 0.25 * PDFt4(x, m, st))
    elif (target == 'Gamma'):
        xL = 0
        a = 4; b = 2
        lnp = (-b*x + (a-1)*np.log(x)) if (x >= xL) else -500
    elif (target == 'Uniform'):
        xL = -np.sqrt(3); xU = np.sqrt(3)
        lnp = 0 if ((x >= xL) & (x<= xU)) else -500
    return (lnp)

def proposal(x, xnew, s, kernel):
    mu_star = 0.1
    if (kernel == 'Uniform'):
        p = 1/(s*np.sqrt(12)) if (abs(x-xnew) <= (s*np.sqrt(12)/2)) else 0
    elif (kernel == 'Gaussian'):
        p = np.exp(-(xnew-x)*(xnew-x)/(2*s*s))/(np.sqrt(2*np.pi)*s)
    elif (kernel == 'Bactrian'):
        m = 0.95
        p1 = 1/(2*s*np.sqrt(2*np.pi*(1-m*m)))
        p2 = np.exp(-(xnew-x+m*s)*(xnew-x+m*s)/(2*(1-m*m)*s*s))
        p3 = np.exp(-(xnew-x-m*s)*(xnew-x-m*s)/(2*(1-m*m)*s*s))
        p = p1 * (p2 + p3)
    elif (kernel == 'Box'):
        a = 0.5; b = 1.43
        p = 1/(2*s*(b-a)) if ((abs(xnew-x) <= s*b) & (abs(xnew-x) >= s*a)) else 0
    elif (kernel == 'Airplane'):
        a = 1; b = 1.47
        if ((abs(xnew-x)<= (s*b)) & (abs(xnew-x) >= (s*a))):
            p = 1/(s*(2*b-a))
        elif (abs(xnew-x) < (s*a)):
            p = 1/(s*s*a*(2*b-a))*abs(xnew-x)
        else:
            p = 0
    elif (kernel == 'StrawHat'):
        a = 1; b = 1.35
        dis = abs(xnew-x)
        if ((dis <= s*b) & (dis >= s*a)):
            p = 3/(2*s*(3*b-2*a))
        elif (dis < s*a):
            p = 3/(2*a*a*s*s*s*(3*b-2*a))*dis*dis
        else:
            p = 0
    elif (kernel == 'MirrorN'):
        NewCenter = 2 * mu_star - x
        p = np.exp(-(xnew-NewCenter)*(xnew-NewCenter)/(2*s*s))/(np.sqrt(2*np.pi)*s)
    elif (kernel == 'MirrorU'):
        NewCenter = 2 * mu_star - x    
        w = s * np.sqrt(12)
        p = 1/w if (abs(NewCenter-xnew) <= w/2) else 0
    elif (kernel == 'MirrorU_Gamma'):
        w = s * np.sqrt(12)
        z = np.exp(w/2)
#         p = 1/(w*xnew) if ((xnew>=x/z) & (xnew <= x*z)) else 0
        p = 1/(w*xnew) if ((xnew>=np.exp(2*0.405-np.log(x))/z) & (xnew <= np.exp(2*0.405-np.log(x))*z)) else 0
    elif (kernel == 'MirrorN_Gamma'):
        xold = 2*0.405-np.log(x)
        p = np.exp(-(xold-np.log(xnew))*(xold-np.log(xnew))/(2*s*s))/(np.sqrt(2*np.pi)*s*xnew) if xnew > 0 else 0
    elif (kernel == 'MirrorU_Uniform'):
        a = -np.sqrt(3); b=-a
        w = s * np.sqrt(12)
        z = np.exp(w/2)
        logx = np.log((x-a)/(b-x))
        logxnew = (xnew-a)/(b-xnew)
        derivative = (b-a)/((xnew-a) * (b-xnew))
        p = 1/w*derivative if ((logxnew>=np.exp(2*0.116-logx)/z) & (logxnew <= np.exp(2*0.116-logx)*z)) else 0
    elif (kernel == 'MirrorN_Uniform'):
        a = -np.sqrt(3); b=-a
        xold = 2*0.116-np.log((x-a)/(b-x))
        logxnew = np.log((xnew-a)/(b-xnew))
        derivative = (b-a)/((xnew-a) * (b-xnew))
        p = np.exp(-(xold-logxnew)*(xold-logxnew)/(2*s*s))/(np.sqrt(2*np.pi)*s)*derivative if np.exp(logxnew)>0 else 0
    return (p)