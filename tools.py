import numpy as np
from scipy.stats import norm

def PDFt4(x, m, s):
    z = (x-m)/s
    pdf = 3/(4*1.414213562*s)*pow(1 + z*z/2, -2.5)
    return(pdf)

def random_bactrian(size):
    m = 0.95
    z = m + np.random.normal(size=size) * np.sqrt(1-m**2)
    judge = np.random.random(size)
    z[judge<0.5] = -1 * z[judge<0.5]
    return(z)

def random_box(size):
    a=0.5; b=1.43
    y = np.random.uniform(low=a,high=b,size=size)
    judge = np.random.random(size)
    y[judge<0.5] = -1 * y[judge<0.5]
    return(y)
    
def random_airplane(size):
    a=1; b=1.47; y =np.zeros(size)
    mu1 = np.random.random(size); mu2 = np.random.random(size); mu3 = np.random.random(size)
    y[mu1<(a/(2*b-a))] = a* np.sqrt(mu2[mu1<(a/(2*b-a))])
    y[mu1>(a/(2*b-a))] = np.random.uniform(low=a, high=b, size=len(y[mu1>(a/(2*b-a))]))
    y[mu3<0.5] = -y[mu3<0.5]
    return(y)
    
def random_strawhat(size):
    a=1; b=1.35; y =np.zeros(size)
    mu1 = np.random.random(size); mu2 = np.random.random(size); mu3 = np.random.random(size)
    y[mu1<(a/(3*b-2*a))] = a* pow(mu2[mu1<(a/(3*b-2*a))],1/3)
    y[mu1>(a/(3*b-2*a))] = np.random.uniform(low=a, high=b, size=len(y[mu1>(a/(3*b-2*a))]))
    y[mu3<0.5] = -y[mu3<0.5]
    return(y)

def rho_k(Y,lag):
    n = len(Y); x=Y.copy()
    x -= np.mean(x)
    x /= np.std(x)
    rho = np.sum(np.multiply(x[0:n-lag],x[lag:n]))
    rho /= n-1
    return(rho)
    
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

def Eff_quantile (Y,p):
    rho0=0; n = len(Y); 
#     quantile = scipy.stats.norm.ppf(percent,loc=mu,scale=std)
    quantile = np.quantile(Y,p)
    x = np.where(Y<=quantile,1,0);Tint=np.var(x);
    for irho in range(1,(n-10)):
        rho = np.cov(x[0:n-irho], x[irho:n])[0,1]
        if ((irho>10) & (rho+rho0<0)):
            break;
        Tint += rho*2;
        rho0 = rho;
#         print('lag:',irho,'|',rho,'sum',':')
    return (p*(1-p)/Tint);

def Eff_quantile_BM (Y,p):
    n = len(Y); length = np.floor(np.sqrt(n)); remainder = n % length;
    y_short = Y[:int(n-remainder)].copy(); batch = len(y_short)/length
    quantile = np.quantile(Y,p)
    x = np.where(y_short<=quantile,1,0)
    split = np.split(x,batch)
    y = np.mean(split,axis=1)
    mu_hat = np.mean(y)
    var = batch*np.sum(np.square(y-mu_hat))/(length-1)
    return (p*(1-p)/var);

def reflect(x,xL,xU):
    n=0; side=0; e=0
    if (x<xL):
        e = xL-x; side=0 
    elif(x>xU):
        e = x-xU; side=1
    n = int(e/(xU-xL));
    if(n%2):
        side = 1-side
    e -= n*(xU-xL)
    x = xU-e if side else xL+e
    return(x)

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
