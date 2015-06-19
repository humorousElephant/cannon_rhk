# Racunam labele izbrane zvezde na podlagi natreniranega modela s train_v1.py

import numpy as np
import pickle
import emcee
import sys
import matplotlib.pylab as plt

# Output filenames
output_folder = ''
#~ results_file = '%strain_v1_theta.txt'%output_folder
#~ l0_output_file = '%sl0.txt'%output_folder
results_file = 'results/train_v1_no_0_star_niter_2000.txt'
l0_output_file = 'data/l0.txt'

# Input data; nekaj je tudi ponovljenih opazovanj, ki so neodvisno na tem seznamu
data = np.loadtxt('data/data.txt', comments='#', delimiter=';') # B, Berr, V, Verr, J, Jerr, K, Kerr, EWirt, eEwirt, R'HK, eR'HK
f = open('data/fluxes.pkl', 'r')
fluxes = pickle.load(f) # Samo fluxi, valovne dolzine so spravljene posebej v wavelengths.txt
f.close()

sigmanL = 0.01 # Izmisljeno

thetaText=['Baseline', 'V-K', 'J-K', "log R'HK", '(V-K)*(V-K)', '(V-K)*(J-K)', "(V-K)*log R'HK", '(J-K)*(J-K)', "(J-K)*log R'HK", "log R'HK*log R'HK", 's']

def prior(thetaL):
	for x in thetaL:
		if np.abs(x)>10: # Vrednosti so majhne, gre za magnitude, pri log R'HK so vrednosti okoli -5 (oz. manj, ker je odsteto povprecje)
			return -np.inf
	return 0.0

def lnp(thetaL, flux, l):
	sL=l[:,-1]
	thetaL = np.array([1]+list(thetaL)) # Dodam 1, ki se mnozi z baseline theta vrednostjo

	S=0.0
	for i, f in enumerate(flux):
		S += (f - l[i,:-1].dot(thetaL))**2 / (sL[i]**2+sigmanL**2) + np.log((sL[i]**2+sigmanL**2))

	S=-0.5*S

	return S+prior(thetaL)

def find_labels(f, theta):
	ndim=theta.shape[1]-2
	nwalkers=ndim*2 * 2
	niter=1000 *2#* 5#*4

	p0=np.random.rand(nwalkers, ndim)
	
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp, args=[f, theta], threads=8)
	pos, prob, _ = sampler.run_mcmc(p0, niter)

	# ESTIMATE PARAMS
	chain=sampler.chain
	lnprob=sampler.lnprobability
	
	max_lnprob_index = np.argmax(lnprob)
	max_lnprob = lnprob[max_lnprob_index/lnprob.shape[1], max_lnprob_index%lnprob.shape[1]]
	most_probable_params = chain[max_lnprob_index/lnprob.shape[1], max_lnprob_index%lnprob.shape[1],:]
	best_step = max_lnprob_index%lnprob.shape[1]
	
	return most_probable_params, best_step, chain, lnprob, nwalkers, ndim, niter
	
def plot(most_probable_params, best_step, chain, lnprob, nwalkers, ndim, niter):
	# PLOT WALKERS
	fig=plt.figure()
	Y_PLOT=2
	X_PLOT=5
	alpha=0.1
	
	# Walkers
	burnin=0.5*niter
	print 'burnin', burnin
	for n in range(ndim):
		ax=fig.add_subplot(Y_PLOT+1,X_PLOT,n+1)
		for i in range(nwalkers): # Risem za posamezne walkerje
			d=np.array(chain[i][burnin:,n])
			ax.plot(d, color='black', alpha=alpha)
		ax.set_xlabel(thetaText[n+1])
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

def format_labels(lab):
	#~ return '; '.join(['%.5f'%x for x in lab[:-2]])
	return '; '.join(['%.5f'%x for x in lab])

if __name__ == "__main__":
	index=0 # index izkljucene zvezde na seznamu, za katero racunam lable

	# model
	theta=np.loadtxt(results_file, delimiter=';')
	theta=theta[:,1:] # prvi stolpec so zaporedne stevilke

	# Izracunaj lable
	labels, best_step, chain, lnprob, nwalkers, ndim, niter = find_labels(fluxes[index], theta)
	
	l0=np.loadtxt(l0_output_file, comments='#') # Povprecje
	labels+=l0
	
	x=data[index]
	truth=np.array([x[2]-x[6], x[4]-x[6], x[10], (x[2]-x[6])*(x[2]-x[6]), (x[2]-x[6])*(x[4]-x[6]), (x[2]-x[6])*x[10], (x[4]-x[6])*(x[4]-x[6]), (x[4]-x[6])*x[10], x[10]*x[10]]) # Pravi labli iz literature; baseline spustim, tega ne modeliram, ampak avtomatsko postavim na 1

	diff = labels-truth
	
	print thetaText[1:-1]
	print 'Labels from this algorithm:'
	print format_labels(labels)
	print 'True labels:'
	print format_labels(truth)
	print 'This-True:'
	print format_labels(diff)
	
	plot(labels, best_step, chain, lnprob, nwalkers, ndim, niter)
