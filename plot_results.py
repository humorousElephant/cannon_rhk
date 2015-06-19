import numpy as np
import scipy.stats
import matplotlib.pylab as plt
from matplotlib import cm
import numpy.ma as ma
import pickle

# Results: filenames
#~ theta_file = 'train_v1_theta.txt'
#~ l0_file = 'l0.txt'
theta_file = 'results/train_v1_no_0_star_niter_2000.txt'
#~ theta_file = 'results/train_v1_theta_niter4000.txt'
l0_file = 'data/l0.txt'

# Data: filenames
wvl_file = 'data/wavelengths.txt'
labels_file = 'data/data.txt'
fluxes_file = 'data/fluxes.pkl'

def param_sensitivity():
	p = np.loadtxt(theta_file, delimiter=';', comments='#')
	w = np.loadtxt(wvl_file)
	
	plt.plot(w, p[:,1], lw=2, c='k', label='Baseline')
	plt.plot(w, p[:,2]+1, c='b', label='V-K')
	plt.plot(w, p[:,3]+1, c='g', label='J-K')
	plt.plot(w, p[:,4]+1, c='r', label="log R'HK")
	plt.legend()
	plt.show()	

def reconstruct_original_fluxes():
	# primerjaj originalne vrednosti fluxov z modelskimi in naredi 2d histogram

	data = np.loadtxt(labels_file, comments='#', delimiter=';') # B, Berr, V, Verr, J, Jerr, K, Kerr, EWirt, eEwirt, R'HK, eR'HK
	f = open(fluxes_file, 'r')
	fluxes = pickle.load(f) # Samo fluxi, valovne dolzine so spravljene posebej v wavelengths.txt
	f.close()
	
	theta=np.loadtxt(theta_file, delimiter=';')
	theta=theta[:,1:] # prvi stolpec so zaporedne stevilke, zadnji pa 's'
	l0=np.loadtxt(l0_file) # Povprecje
	
	labels=np.array([[1, x[2]-x[6], x[4]-x[6], x[10], (x[2]-x[6])*(x[2]-x[6]), (x[2]-x[6])*(x[4]-x[6]), (x[2]-x[6])*x[10], (x[4]-x[6])*(x[4]-x[6]), (x[4]-x[6])*x[10], x[10]*x[10]] for x in data])

	for i in range(labels.shape[1]-1):
		labels[:,i+1] -= l0[i]

	# Vkljucena je tudi 0. zvezda, ki pa jo za training izpustim

	h=[]
	bins=np.linspace(-0.1, 0.1, 40)
	for i in range(fluxes.shape[1]):
		f=fluxes[:,i]
		hist, edges = np.histogram(f-labels.dot(theta[i,:-1]), bins=bins) # , range=(3500, 7000)
		h.append(hist)
	
	hist=np.array(h)

	# PLOT
	fig=plt.figure()
	ax=fig.add_subplot(111)
	hist = ma.array(hist, mask=0)
	X,Y = np.meshgrid(range(len(h)), bins)
	cm.Greys.set_bad('white', alpha=None)
	image = ax.pcolormesh(X,Y,hist.T, cmap=cm.Greys) # vmin=-0.3, vmax=0.5
	ax.set_ylim(bins[0], bins[-1])
	ax.set_xlim(0, len(hist))

	# Cez narisi se baseline
	ax.plot(range(len(h)), (theta[:,0]-1)/10.0, c='k', label='Baseline')
	ax.plot(range(len(h)), theta[:,-1], c='r', label=r'$s_\lambda$')
	ax.legend()
	
	ax.set_xlabel('Pixel')
	ax.set_ylabel('(Cannon Model Flux) - (True Flux)')
	
	plt.show()	

def plot(nwalkers, niter, ndim, chain, lnprob, thetaText, best_step):
	# PLOT WALKERS
	fig=plt.figure()
	Y_PLOT=3
	X_PLOT=4
	alpha=0.1
	
	# Walkers
	burnin=0.5*niter # burnin za risanje
	
	# Posamezni walkerji
	for n in range(ndim):
		ax=fig.add_subplot(Y_PLOT+1,X_PLOT,n+1)
		for i in range(nwalkers): # Risem za posamezne walkerje
			d=np.array(chain[i][burnin:,n])
			ax.plot(d, color='black', alpha=alpha)
		ax.set_xlabel(thetaText[n])
		if best_step-burnin>0:
			plt.axvline(x=best_step-burnin, linewidth=1, color='red')
		
	# Lnprob
	ax=fig.add_subplot(Y_PLOT+1,1,Y_PLOT+1)
	for i in range(nwalkers):
		ax.plot((lnprob[i][burnin:]), color='black', alpha=alpha)
	plt.axvline(x=best_step-burnin, linewidth=1, color='red')
	
	#~ import triangle
	#~ fig = triangle.corner(sampler.flatchain, truths=most_probable_params) # labels=thetaText
	
	plt.show()	

if __name__ == "__main__":
	#~ param_sensitivity()
	reconstruct_original_fluxes()
