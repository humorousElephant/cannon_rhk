# Racunam labele izbrane zvezde na podlagi natreniranega modela s train_v1.py

import numpy as np
import pickle
import emcee
import sys

# Output filenames
output_folder = ''
#~ results_file = 'results/train_v1_theta_niter4000.txt'
results_file = 'results/train_v1_theta_niter4000_data2.txt'
#~ results_file = 'results/train_v1_theta_niter4000_vmk.txt'
l0_output_file = 'data/l0.txt'

# Input data; nekaj je tudi ponovljenih opazovanj, ki so neodvisno na tem seznamu
data = np.loadtxt('data/data2.txt', comments='#', delimiter=';') # B, Berr, V, Verr, J, Jerr, K, Kerr, EWirt, eEwirt, R'HK, eR'HK
f = open('data/fluxes2.pkl', 'r')
fluxes = pickle.load(f) # Samo fluxi, valovne dolzine so spravljene posebej v wavelengths.txt
f.close()

sigmanL = 0.01 # Izmisljeno

thetaText=['Baseline', 'V-K', 'J-K', "log R'HK", '(V-K)*(V-K)', '(V-K)*(J-K)', "(V-K)*log R'HK", '(J-K)*(J-K)', "(J-K)*log R'HK", "log R'HK*log R'HK", 's']

def prior(thetaL):
	for x in thetaL:
		if np.abs(x)>10: # Vrednosti so majhne
			return -np.inf
	return 0.0

def lnp(l, flux, theta):
	labels = np.array([1, l[0], l[1], l[2], l[0]*l[0], l[0]*l[1], l[0]*l[2], l[1]*l[1], l[1]*l[2], l[2]*l[2]])
	
	S=-0.5*np.sum([(f - t[:-1].dot(labels))**2 / (t[-1]**2+sigmanL**2) + np.log((t[-1]**2+sigmanL**2)) for f, t in zip(flux, theta)])

	return S+prior(labels)

def find_labels(f, theta, pool=None):
	ndim=3
	nwalkers=10
	niter=400

	p0=np.random.rand(nwalkers, ndim)
	
	if pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp, args=[f, theta], pool=pool)
	else:
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

def find_labels_for_1_spectrum():
	index=0 # index izkljucene zvezde na seznamu, za katero racunam lable

	# model
	theta=np.loadtxt(results_file, delimiter=';')
	theta=theta[:,1:] # prvi stolpec so zaporedne stevilke

	# Izracunaj lable
	labels, best_step, chain, lnprob, nwalkers, ndim, niter = find_labels(fluxes[index], theta)
	
	l0=np.loadtxt(l0_output_file, comments='#') # Povprecje
	labels+=l0[:len(labels)]
	
	x=data[index]
	truth=np.array([x[2]-x[6], x[4]-x[6], x[10]]) # Pravi labli iz literature; baseline spustim, tega ne modeliram, ampak avtomatsko postavim na 1

	diff = labels-truth
	
	print thetaText[1:-1]
	print 'Labels from this algorithm:'
	print format_labels(labels)
	print 'True labels:'
	print format_labels(truth)
	print 'This-True:'
	print format_labels(diff)
	
	plot(labels, best_step, chain, lnprob, nwalkers, ndim, niter, l0)
	
def find_labels_for_all_spectra():
	# model
	theta=np.loadtxt(results_file, delimiter=';')
	theta=theta[:,1:] # prvi stolpec so zaporedne stevilke
	l0=np.loadtxt(l0_output_file, comments='#') # Povprecje

	POOL = emcee.utils.MPIPool()
	if not POOL.is_master():
		POOL.wait()
		sys.exit(0)

	f1 = open('results/find_labels_data2.txt', 'wb')

	for i, f in enumerate(fluxes):
		labels, best_step, chain, lnprob, nwalkers, ndim, niter = find_labels(f, theta, pool=POOL)
		labels+=l0[:len(labels)]

		x=data[i]
		truth=np.array([x[2]-x[6], x[4]-x[6], x[10]]) # Pravi labli iz literature; baseline spustim, tega ne modeliram, ampak avtomatsko postavim na 1
		
		line = '; '.join(['%.5f'%x for x in labels])
		line += '; ' + '; '.join(['%.5f'%x for x in truth])
		print i, line
		f1.write(line+'\n')

	f1.close()
	POOL.close()
	
def plot(most_probable_params, best_step, chain, lnprob, nwalkers, ndim, niter, l0):
	import matplotlib.pylab as plt
	# PLOT WALKERS
	fig=plt.figure()
	Y_PLOT=1
	X_PLOT=3
	alpha=0.1
	
	# Walkers
	burnin=0.1*niter
	print 'burnin', burnin
	for n in range(ndim):
		ax=fig.add_subplot(Y_PLOT+1,X_PLOT,n+1)
		for i in range(nwalkers): # Risem za posamezne walkerje
			d=np.array(chain[i][burnin:,n])
			ax.plot(d+l0[n], color='black', alpha=alpha) # PRISTEJEM l0
		ax.set_xlabel(thetaText[n+1])
		if best_step-burnin>0:
			plt.axvline(x=best_step-burnin, linewidth=1, color='red')
		
	# Lnprob
	ax=fig.add_subplot(Y_PLOT+1,1,Y_PLOT+1)
	for i in range(nwalkers):
		ax.plot((lnprob[i][burnin:]), color='black', alpha=alpha)
	if best_step-burnin>0:
		plt.axvline(x=best_step-burnin, linewidth=1, color='red')
	
	#~ import triangle
	#~ fig = triangle.corner(sampler.flatchain, truths=most_probable_params) # labels=thetaText
	
	plt.show()

def format_labels(lab):
	return '; '.join(['%.5f'%x for x in lab])

if __name__ == "__main__":
	#~ find_labels_for_1_spectrum()
	find_labels_for_all_spectra()
