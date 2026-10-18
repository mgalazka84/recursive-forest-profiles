"""Reproduce the figures and independently check finite-size moment formulas.

Run: python3 verify_and_figures.py
Requires numpy, scipy, matplotlib. No network access is used.
"""
from pathlib import Path
from math import factorial
from fractions import Fraction
import json
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
RESULTS = {}


def immigration(n, alpha, theta=0.65, scheme='power'):
    if scheme == 'deleted_N':
        return 2 / (n * (n + 1))
    if scheme == 'deleted_O':
        return 2 / (n + 1)
    return min(1.0, theta * n ** (-alpha))


def birth_matrix(n, h, model, beta, rn):
    """p = rn*e0 + B*X for X=(L_0,...,L_H,W_0,...,W_H)."""
    size = h + 1
    B = np.zeros((size, 2 * size))
    if model == 'N':
        c = 2 * (1 - rn) / ((n - 1) * (n + 2 * beta))
        for k in range(1, size):
            B[k, size + k - 1] = c
        v = n + beta
    else:
        d = 2 * (1 - rn) / (n * (n - 1))
        for k in range(1, size):
            B[k, k - 1] = n * d
            B[k, size + k - 1] = -d
        v = n
    D = np.vstack((np.eye(size), v * np.eye(size)))
    return B, D


def moment_step(mean, cov, n, h, model, beta, rn):
    B, D = birth_matrix(n, h, model, beta, rn)
    ep = B @ mean
    ep[0] += rn
    cross = cov @ B.T @ D.T
    updated = cov + cross + cross.T + D @ (np.diag(ep) - np.outer(ep, ep)) @ D.T
    return mean + D @ ep, updated, ep


def scalar_cov_step(cov, ep, n, h, model, beta, rn):
    """The manuscript's three scalar systems, implemented separately."""
    size = h + 1
    S, X, P = cov[:size, :size], cov[:size, size:], cov[size:, size:]
    Sn, Xn, Pn = S.copy(), X.copy(), P.copy()
    eta = np.diag(ep) - np.outer(ep, ep)

    def get(M, r, s):
        return M[r, s] if r >= 0 and s >= 0 else 0.0

    for r in range(size):
        for s in range(size):
            e = eta[r, s]
            if model == 'N':
                c = 2 * (1 - rn) / ((n - 1) * (n + 2 * beta))
                v = n + beta
                Sn[r, s] += c * (get(X, r, s-1) + get(X, s, r-1)) + e
                Xn[r, s] += v*c*get(X, r, s-1) + c*get(P, r-1, s) + v*e
                Pn[r, s] += v*c*(get(P, r, s-1) + get(P, r-1, s)) + v*v*e
            else:
                d = 2 * (1 - rn) / (n * (n - 1))
                Sn[r, s] += d*(n*get(S, r, s-1)-get(X, r, s-1)) + d*(n*get(S, s, r-1)-get(X, s, r-1)) + e
                Xn[r, s] += n*d*(n*get(S, r, s-1)-get(X, r, s-1)) + d*(n*get(X, r-1, s)-get(P, r-1, s)) + n*e
                Pn[r, s] += n*d*(n*get(X, s-1, r)-get(P, r, s-1)) + n*d*(n*get(X, r-1, s)-get(P, r-1, s)) + n*n*e
    return np.block([[Sn, Xn], [Xn.T, Pn]])


def exhaustive_checks():
    H = 3
    cases = [(model, beta, alpha, 'power')
             for model, beta in [('N', -0.75), ('N', 0.0), ('N', 1.5), ('O', 0.0)]
             for alpha in [0.55, 1.0, 1.7]]
    cases += [('N', 1.0, 2.0, 'deleted_N'), ('O', 0.0, 1.0, 'deleted_O')]
    max_mean_error = max_cov_error = max_system_error = 0.0
    comparisons = configurations = 0
    for model, beta, alpha, scheme in cases:
        mean = np.zeros(2*(H+1)); mean[0] = 1; mean[H+1] = 1 + (beta if model == 'N' else 0)
        cov = np.zeros((len(mean), len(mean)))
        states = [((0,), 1.0)]
        for n in range(2, 8):
            rn = immigration(n, alpha, scheme=scheme)
            mn, cn, ep = moment_step(mean, cov, n, H, model, beta, rn)
            cs = scalar_cov_step(cov, ep, n, H, model, beta, rn)
            max_system_error = max(max_system_error, float(np.max(np.abs(cs-cn))))
            if model == 'N':
                q = np.array([2*(i+beta)/((n-1)*(n+2*beta)) for i in range(1, n)])
            else:
                q = np.array([2*(n-i)/(n*(n-1)) for i in range(1, n)])
            next_states = []
            em = np.zeros_like(mean); raw = np.zeros_like(cov); mass = 0.0
            for heights, probability in states:
                for parent in range(n):
                    prob = probability * (rn if parent == 0 else (1-rn)*q[parent-1])
                    hs = heights + ((0 if parent == 0 else heights[parent-1]+1),)
                    next_states.append((hs, prob))
                    arr = np.array(hs)
                    labels = np.arange(1, n+1) + (beta if model == 'N' else 0)
                    L = np.array([np.count_nonzero(arr == k) for k in range(H+1)])
                    W = np.array([labels[arr == k].sum() for k in range(H+1)])
                    z = np.concatenate((L, W))
                    mass += prob; em += prob*z; raw += prob*np.outer(z,z)
            ec = raw - np.outer(em, em)
            assert abs(mass-1) < 2e-12
            max_mean_error = max(max_mean_error, float(np.max(np.abs(em-mn))))
            max_cov_error = max(max_cov_error, float(np.max(np.abs(ec-cn))))
            configurations += len(next_states); comparisons += 1
            states = next_states; mean, cov = mn, cn
    assert max_mean_error < 2e-10 and max_cov_error < 2e-9 and max_system_error < 2e-10
    RESULTS['exhaustive_small_forests'] = dict(cases=len(cases), comparisons=comparisons,
        parent_configurations=configurations, n_max=7, levels='0..3',
        max_mean_absolute_error=max_mean_error, max_covariance_absolute_error=max_cov_error,
        max_scalar_vs_matrix_absolute_error=max_system_error)


def triangular_constants(p, H, model, theta=0.65):
    A = np.zeros((H+1, H+1)); B = A.copy(); D = A.copy()
    lam = np.array([theta*(2/(p+1))**h if model=='N' else theta*(2/(p*(p+1)))**h for h in range(H+1)])

    def g(M,r,s):
        return M[r,s] if r>=0 and s>=0 else 0.0

    for t in range(2*H+1):
        for r in range(H+1):
            s=t-r
            if s<0 or s>H: continue
            e=lam[r] if r==s else 0
            if model=='N':
                D[r,s]=(2*g(D,r,s-1)+2*g(D,r-1,s)+e)/(p+2)
                B[r,s]=(2*g(B,r,s-1)+2*g(D,r-1,s)+e)/(p+1)
                if p>0: A[r,s]=(2*g(B,r,s-1)+2*g(B,s,r-1)+e)/p
                else: A[r,s]=2*g(B,r,s-1)+2*g(B,s,r-1)+e
            else:
                A[r,s]=(2*(g(A,r,s-1)-g(B,r,s-1))+2*(g(A,s,r-1)-g(B,s,r-1))+e)/p
                B[r,s]=(2*(g(A,r,s-1)-g(B,r,s-1))+2*(g(B,r-1,s)-g(D,r-1,s))+e)/(p+1)
                D[r,s]=(2*(g(B,s-1,r)-g(D,r,s-1))+2*(g(B,r-1,s)-g(D,r-1,s))+e)/(p+2)
    return A,lam


def analytic_checks():
    # Explicit critical covariance versus the triangular mixed-moment system.
    A,_=triangular_constants(0.0, 7, 'N')
    explicit=np.array([[0.65*(2**(r+s+1)-2**max(r,s)) for s in range(8)] for r in range(8)])
    critical_error=float(np.max(np.abs(A-explicit)))
    assert np.allclose(A, explicit, rtol=1e-12, atol=1e-12)

    # Subcritical integral covariance versus both triangular systems.
    # Inverse transforms of 2^h/[s^(h+1)(s+1)^h], for h=0,...,3.
    older=[lambda t:1.0,
           lambda t:2*(t-1+np.exp(-t)),
           lambda t:2*t*t-8*t+12-np.exp(-t)*(12+4*t),
           lambda t:(4/3)*t**3-12*t*t+48*t-80+np.exp(-t)*(80+32*t+4*t*t)]
    newer=[lambda t: 1.0] + [lambda t,h=h: 2**h*(1-np.exp(-t)*sum(t**k/factorial(k) for k in range(h))) for h in range(1,4)]
    max_relative=0.0; comparisons=0
    for model, funcs in [('N',newer),('O',older)]:
        for p in [0.25, 0.6, 0.9]:
            A,lam=triangular_constants(p,3,model)
            for r in range(4):
                for s in range(4):
                    val=sum(lam[k]*quad(lambda t,k=k: np.exp(-p*t)*funcs[r-k](t)*funcs[s-k](t),0,np.inf,epsabs=1e-7,epsrel=2e-10)[0] for k in range(min(r,s)+1))
                    max_relative=max(max_relative,abs(val-A[r,s])/max(1,abs(A[r,s])))
                    comparisons+=1
    assert max_relative < 2e-8

    # Exact root-deletion conditional kernels, checked as rational identities.
    for n in range(2,20):
        for i in range(1,n):
            p_new=Fraction(2*(i+1),n*(n+1)); rn=Fraction(2,n*(n+1))
            assert p_new/(1-rn)==Fraction(2*(i+1),(n-1)*(n+2))
            p_old=Fraction(2*(n-i),n*(n+1)); rn=Fraction(2,n+1)
            assert p_old/(1-rn)==Fraction(2*(n-i),n*(n-1))
    for h in range(1,12):
        assert (2**(2*h+1)-2**h)+2*(2**(2*h+2)-2**(h+1))+(2**(2*h+3)-2**(h+1))==18*4**h-7*2**h
    RESULTS['analytic_constants']=dict(critical_matrix_size=8, critical_max_absolute_error=critical_error,
        subcritical_integral_comparisons=comparisons, subcritical_max_relative_error=max_relative,
        root_deletion_rational_identities='passed',degree_sum_constants='passed')


def exact_means(nmax,H,model,alpha,beta=0,theta=1):
    n=np.arange(nmax+1,dtype=float)
    rn=np.zeros(nmax+1); rn[1]=1; rn[2:]=np.minimum(1,theta*n[2:]**(-alpha))
    L=np.zeros((H+1,nmax+1)); W=np.zeros_like(L)
    L[0]=np.cumsum(rn)
    weights=n+(beta if model=='N' else 0)
    W[0]=np.cumsum(weights*rn)
    c=np.zeros_like(n)
    if model=='N': c[2:]=2*(1-rn[2:])/((n[2:]-1)*(n[2:]+2*beta))
    else: c[2:]=2*(1-rn[2:])/(n[2:]*(n[2:]-1))
    for h in range(1,H+1):
        inc=np.zeros_like(n)
        if model=='N': inc[2:]=c[2:]*W[h-1,1:-1]
        else: inc[2:]=c[2:]*(n[2:]*L[h-1,1:-1]-W[h-1,1:-1])
        L[h]=np.cumsum(inc); W[h]=np.cumsum(weights*inc)
    return L,W


def appendix_generating_functions():
    max_error=0
    for nmax in range(1,13):
        H=12; mean=np.zeros(2*(H+1)); mean[0]=1; mean[H+1]=2
        cov=np.zeros((len(mean),len(mean)))
        for n in range(2,nmax+1):
            mean,cov,_=moment_step(mean,cov,n,H,'N',1.0,2/(n*(n+1)))
        P=[Fraction(1)]
        for j in range(1,nmax+1):
            Q=[Fraction(0)]*(len(P)+1)
            for k,c in enumerate(P):
                Q[k]+=c; Q[k+1]+=Fraction(2,j)*c
            P=Q
        numerator=[Fraction(2,nmax+1)*c for c in P]
        numerator[0]-=1; numerator[1]-=2
        A=[Fraction(0)]*nmax
        for k in range(nmax,0,-1):
            A[k-1]=numerator[k]/2
            numerator[k]-=2*A[k-1]; numerator[k-1]+=A[k-1]
        assert all(c==0 for c in numerator)
        A[0]+=1
        G=P[1:]
        for h in range(H+1):
            ah=float(A[h]) if h<len(A) else 0
            gh=float(G[h]) if h<len(G) else 0
            max_error=max(max_error,abs(ah-mean[h]),abs(gh-mean[H+1+h]))
    assert max_error<1e-9
    RESULTS['appendix_A_generating_functions']=dict(n_max=12, max_absolute_error=max_error)


def general_response_checks():
    """Check the new response statement using exact discrete path sums.

    Track generation 1 and 2 descendants of one marked vertex. Parent
    normalization is computed directly for each insertion, independently
    of the continuum convolution used for the limiting values.
    """
    slow_integral = quad(lambda x: 1 / (1 - np.log(x)), 0, 1,
                         epsabs=1e-12, epsrel=1e-12)[0]
    a = 1 / slow_integral
    kernels = {
        'power_3x2': (
            lambda x: 3*x*x,
            lambda u: 1.5*(1-np.exp(-2*u)),
            lambda u: 2.25*(1-np.exp(-2*u)*(1+2*u))),
        'positive_endpoint_half_plus_x': (
            lambda x: .5+x,
            lambda u: .5*u+1-np.exp(-u), None),
        'zero_endpoint_unbounded_response': (
            lambda x: a/(1-np.log(x)),
            lambda u: a*np.log1p(u), None)
    }
    report = {}
    u = np.log(4)
    for name, (f, g1, g2) in kernels.items():
        target2 = (g2(u) if g2 else
                   quad(lambda v: float(f(np.exp(-v)))*g1(u-v), 0, u,
                        epsabs=1e-12, epsrel=1e-12)[0])
        target = np.array([g1(u), target2])
        errors = []
        rows = []
        for seed in [200, 800]:
            last = 4*seed
            marked1 = np.zeros(last+1)
            k = np.zeros(2)
            for n in range(seed+1, last+1):
                labels = np.arange(1, n, dtype=float)
                weights = f(labels/n)
                denominator = float(np.sum(weights))
                arrival = 1-1/n
                birth1 = arrival*float(f(seed/n))/denominator
                birth2 = arrival*float(np.dot(weights[seed:], marked1[seed+1:n]))/denominator
                marked1[n] = birth1
                k += (birth1, birth2)
            error = np.abs(k-target)
            rows.append(dict(seed=seed, horizon=last, response=k.tolist(),
                             absolute_errors=error.tolist()))
            errors.append(error)
        assert np.all(errors[1] < errors[0]), (name, errors)
        assert np.max(errors[1]) < .006, (name, errors)
        report[name] = dict(continuum_response=target.tolist(), discrete_checks=rows)
    # The Mellin/Laplace identity for a third kernel, beyond the linear pair.
    transform_error = 0.0
    for h in [1, 2]:
        gh = kernels['power_3x2'][h]
        for z in [.4, 1.1, 2.0]:
            numerical = quad(lambda v: np.exp(-z*v)*gh(v), 0, np.inf,
                             epsabs=1e-11, epsrel=1e-11)[0]
            exact = (3/(z+2))**h/z
            transform_error = max(transform_error, abs(numerical-exact))
    assert transform_error < 1e-10
    report['power_kernel_laplace_max_absolute_error'] = transform_error
    report['slow_kernel_normalization'] = a
    report['scope'] = 'Finite-size checks of fixed-ratio continuum response; not a general profile CLT.'
    RESULTS['general_scale_response'] = report


def critical_finite_size_checks():
    nmax = 2_000_000
    L, W = exact_means(nmax, 3, 'O', 1)
    labels = np.arange(1, nmax+1, dtype=float)
    harmonic = np.sum(1/labels)
    harmonic2 = np.sum(1/(labels*labels))
    exact_b1 = harmonic*harmonic+harmonic2-2*harmonic
    residual = abs(exact_b1-L[1,nmax])
    assert residual < 1e-8
    assert abs(W[0,nmax]-nmax) < 1e-8
    lead = np.array([2**h*np.log(nmax)**(h+1)/factorial(h+1) for h in range(4)])
    ratios = L[:,nmax]/lead
    assert np.all(np.abs(ratios-np.array([1.0398,.9456,.7535,.5281])) < .00005)
    RESULTS['critical_finite_size_means'] = dict(
        model='O', immigration='r_n=1/n', n=nmax,
        exact_means=L[:,nmax].tolist(), leading_terms=lead.tolist(),
        ratios=ratios.tolist(), harmonic_identity_absolute_error=residual)


def figures():
    plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix',
        'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    blue='#155C9A'; orange='#BB4D14'
    # A schematic, not a sample from either random attachment model.
    # Both panels start from the same seven-vertex forest.
    base={1:0,2:1,3:1,4:0,5:4,6:2,7:4}
    fig,ax=plt.subplots(1,2,figsize=(6.3,2.7),layout='constrained')
    for a,immigrates in zip(ax,[True,False]):
        parents=dict(base);parents[8]=0 if immigrates else 7
        heights={}
        for vertex,parent in parents.items():
            heights[vertex]=0 if parent==0 else heights[parent]+1
        x={1:.65,2:.15,3:1.15,4:2.9,5:2.4,6:.15,7:3.4,8:5.05 if immigrates else 3.4}
        for h in range(3):
            a.axhline(2-h,color='#dddddd',lw=.7,zorder=0)
        for vertex,parent in parents.items():
            if parent:
                a.plot([x[parent],x[vertex]],[2-heights[parent],2-heights[vertex]],
                       color=orange if vertex==8 else '#708090',lw=1.5,zorder=1)
        for vertex in parents:
            a.scatter(x[vertex],2-heights[vertex],s=190,facecolor=orange if vertex==8 else 'white',
                      edgecolor=orange if vertex==8 else blue,lw=1.4,zorder=3)
            a.text(x[vertex],2-heights[vertex],str(vertex),ha='center',va='center',
                   fontsize=8,color='white' if vertex==8 else '#1c3040',zorder=4)
        counts=tuple(sum(h==k for h in heights.values()) for k in range(3))
        expected=(3,4,1) if immigrates else (2,4,2)
        assert counts==expected and sum(counts)==8
        a.set(xlim=(-.35,5.5),ylim=(-.42,2.55),xticks=[],yticks=[2,1,0],
              yticklabels=['Roots','Children','Grandchildren'],
              title='(a) New root' if immigrates else '(b) New child')
        a.tick_params(axis='both',length=0,labelsize=7)
        for spine in a.spines.values():spine.set_visible(False)
        a.set_xlabel(r'$(L(0),L(1),L(2))='+str(counts).replace(' ','')+'$',fontsize=9)
    fig.savefig(FIG/'forest_growth_schematic.pdf');fig.savefig(FIG/'forest_growth_schematic.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(1,2,figsize=(6.3,2.7),layout='constrained')
    x=np.linspace(0,1,400); u=np.linspace(0,7,400)
    ax[0].plot(x,2*x,color=blue,label=r'$N_\beta$: $2x$',lw=2)
    ax[0].plot(x,2*(1-x),color=orange,ls='--',label=r'$O$: $2(1-x)$',lw=2)
    ax[0].set(xlabel=r'Relative parent label $x=i/n$',ylabel='Limiting density',title='(a) Parent choice',xlim=(0,1),ylim=(0,2.12))
    ax[0].legend(frameon=False,loc='upper center')
    ax[1].plot(u,2*(1-np.exp(-u)),color=blue,label=r'$N_\beta$',lw=2)
    ax[1].plot(u,2*(u-1+np.exp(-u)),color=orange,ls='--',label=r'$O$',lw=2)
    ax[1].set(xlabel=r'Logarithmic age $u=\log(n/j)$',ylabel='Expected direct descendants',title='(b) First-generation response',xlim=(0,7),ylim=(0,12.5))
    ax[1].legend(frameon=False)
    for a in ax:a.grid(alpha=.18)
    fig.savefig(FIG/'attachment_response.pdf'); fig.savefig(FIG/'attachment_response.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(1,3,figsize=(6.3,2.6),layout='constrained')
    sample=np.unique(np.geomspace(10,1_000_000,350).astype(int))
    values={}
    for a,alpha in zip(ax,[.6,1.0,1.5]):
        for model,color,style in [('N',blue,'-'),('O',orange,'--')]:
            L,_=exact_means(1_000_000,1,model,alpha)
            a.loglog(sample,L[1,sample],color=color,ls=style,lw=1.8,label=r'$N_0$' if model=='N' else '$O$')
            values[f'{model}_alpha_{alpha}']={str(int(n)):float(L[1,n]) for n in [100,1000,10000,100000,1000000]}
        a.set(xlabel=r'Forest size $n$',title=rf'$\alpha={alpha:g}$')
        a.grid(alpha=.18,which='major'); a.set_xticks([10**2,10**4,10**6])
    ax[0].set_ylabel(r'Exact mean $\mathbb{E} L_n(1)$')
    ax[0].legend(frameon=False)
    fig.savefig(FIG/'exact_mean_phases.pdf');fig.savefig(FIG/'exact_mean_phases.png',dpi=180);plt.close(fig)
    RESULTS['figure_mean_values']=values

    levels=np.arange(4)
    A=np.array([[2**(r+s+1)-2**max(r,s) for s in levels] for r in levels],dtype=float)
    Cn=A/np.sqrt(np.outer(np.diag(A),np.diag(A)))
    Co=np.array([[np.sqrt((2*r+1)*(2*s+1))/(r+s+1) for s in levels] for r in levels])
    assert np.allclose(Cn,Cn.T) and np.allclose(Co,Co.T)
    assert np.linalg.eigvalsh(Cn).min()>0 and np.linalg.eigvalsh(Co).min()>0
    fig,ax=plt.subplots(1,3,figsize=(6.3,2.3),layout='constrained')
    for a,C,title in zip(ax,[Cn,Co,np.ones((4,4))],[r'$N_\beta$: critical',r'$O$: critical',r'$O$: summable']):
        im=a.imshow(C,vmin=0,vmax=1,cmap='Blues')
        a.set(title=title,xlabel='Level',xticks=levels,yticks=levels)
        for r in levels:
            for s in levels:
                a.text(s,r,f'{C[r,s]:.2f}',ha='center',va='center',fontsize=7,color='white' if C[r,s]>.65 else 'black')
    ax[0].set_ylabel('Level')
    fig.colorbar(im,ax=ax,shrink=.8,label='Correlation',ticks=[0,.5,1])
    fig.savefig(FIG/'limiting_correlations.pdf');fig.savefig(FIG/'limiting_correlations.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    exhaustive_checks()
    analytic_checks()
    appendix_generating_functions()
    general_response_checks()
    critical_finite_size_checks()
    figures()
    (ROOT/'verification_results.json').write_text(json.dumps(RESULTS,indent=2))
    print(json.dumps({k:v for k,v in RESULTS.items() if k!='figure_mean_values'},indent=2))
