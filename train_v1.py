# Test modela
# Natreniram brez izbrane zvezde in nato izracunam njene lable s tem modelom (label_v1.py)

# mpirun -np 8 python train_v1.py

import numpy as np
import random
import pickle
import emcee
import sys

# Output filenames
output_folder = 'results/'
results_file = '%strain_v1_theta_niter4000_data2.txt'%output_folder

# Input data; nekaj je tudi ponovljenih opazovanj, ki so neodvisno na tem seznamu
data_folder = '/home/marusa/rave/cannon_rhk/data'
l0_output_file = '%s/l0.txt'%data_folder
data = np.loadtxt('%s/data2.txt'%data_folder, comments='#', delimiter=';') # B, Berr, V, Verr, J, Jerr, K, Kerr, EWirt, eEwirt, R'HK, eR'HK
f = open('%s/fluxes2.pkl'%data_folder, 'r') # Samo fluxi pri izbranih valovnih dolzinah, valovne dolzine so zapisane posebej v fajlu wavelengths.txt
fluxes = pickle.load(f)
f.close()

# Majhen subset podatkov za hitro testiranje, ce sploh dela
#fluxes=fluxes[:20,:10]
#data=data[:20]

# Izkljuci zvezde, ker imajo prevelike napake v fotometriji
exclude_lines=[35, 106, 190, 191]
#~ for i, ex in enumerate(sorted(exclude_lines)):
	#~ data = np.delete(data,(ex-i), axis=0)
	#~ fluxes = np.delete(fluxes,(ex-i), axis=0)


# Izloci zvezde, ki imajo velik V-K
exclude=[]
for i, x in enumerate(data):
	if x[2]-x[6]>4:
		exclude.append(i)

for i, ex in enumerate(sorted(exclude)):
	data = np.delete(data,(ex-i), axis=0)
	fluxes = np.delete(fluxes,(ex-i), axis=0)

sigmanL = 0.01 # Izmisljeno!!!!!!!!!

thetaText=['Baseline', 'V-K', 'J-K', "log R'HK", '(V-K)*(V-K)', '(V-K)*(J-K)', "(V-K)*log R'HK", '(J-K)*(J-K)', "(J-K)*log R'HK", "log R'HK*log R'HK", 's']

def random_sign():
	x = np.random.rand()
	if x<0.5:
		return -1.0
	else:
		return 1.0

def initial_positions_of_walkers(ndim, nwalkers):
	p0=[]
	i=0
	while i<nwalkers:
		proposed_point = [random.random()*random_sign() for d in range(ndim)]
		#~ proposed_point = np.random.rand(len(ndim))
		if np.isinf(prior(proposed_point)):
			continue
		else:
			p0.append(proposed_point)
			i+=1
	p0=np.array(p0)
	return p0

def prior(thetaL):
	for x in thetaL[1:-1]:
		if np.abs(x)>1: # Koeficienti so majhni
			return -np.inf
	
	if thetaL[-1]<0.0: # variance 's' can't be less than zero
		return -np.inf
	elif thetaL[0]<0 or thetaL[0]>1.1: # baseline spectrum can't be less than zero or far greater than continuum
		return -np.inf
	else:
		return 0.0

def lnp(thetaL, f, l):
	sL=thetaL[-1]
	sf = - 0.5 * np.log((sL**2+sigmanL**2))
	ss = (sL**2+sigmanL**2)
	
	S = -0.5 * np.sum((f - l.dot(np.array(thetaL[:-1])))**2) / ss + sf*float(len(l))

	return S+prior(thetaL)

def train_bayesian_second_order_1PX(f, l, pool=None, threads=8, plot=False, niter=4000): # Single pixel
	ndim=len(l[0])+1
	nwalkers=ndim*2 * 2
	#~ niter=1000 * 4#*4

	p0=initial_positions_of_walkers(ndim, nwalkers)
	#~ p0=np.random.rand(nwalkers, ndim)
	if pool is not None:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp, args=[f, l], pool=pool)
	else:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp, args=[f, l], threads=threads)
	pos, prob, _ = sampler.run_mcmc(p0, niter)

	# ESTIMATE PARAMS: results are the most likely values
	chain=sampler.chain
	lnprob=sampler.lnprobability
	
	max_lnprob_index = np.argmax(lnprob)
	max_lnprob = lnprob[max_lnprob_index/lnprob.shape[1], max_lnprob_index%lnprob.shape[1]]
	most_probable_params = chain[max_lnprob_index/lnprob.shape[1], max_lnprob_index%lnprob.shape[1],:]
	
	# Narisi walkerje
	if plot:
		import plot_results
		plot_results.plot(nwalkers, niter, ndim, chain, lnprob, thetaText, max_lnprob_index%lnprob.shape[1])
	
	return most_probable_params

def train_without_1_star():
	# V-K, J-K, log R'HK
	labels=np.array([[1, x[2]-x[6], x[4]-x[6], x[10], (x[2]-x[6])*(x[2]-x[6]), (x[2]-x[6])*(x[4]-x[6]), (x[2]-x[6])*x[10], (x[4]-x[6])*(x[4]-x[6]), (x[4]-x[6])*x[10], x[10]*x[10]] for x in data])

	# Subtract the mean values of labels
	l0 = [np.mean(labels[:,i+1]) for i in range(labels.shape[1]-1)]
	for i in range(labels.shape[1]-1):
		labels[:,i+1] -= l0[i]

	POOL = emcee.utils.MPIPool()
	if not POOL.is_master():
		POOL.wait()
		sys.exit(0)

	if POOL.rank==0:
		# Print mean label values to file 'l0_output_file'
		f=open(l0_output_file, 'wb')
		f.write('# Mean values of labels\n')
		line='; '.join(thetaText[1:-1])
		f.write('# '+line+'\n')
		line='\n'.join(['%.5e'%x for x in l0])
		f.write(line+'\n')
		f.close()
		
		# Print column names to 'results_file' with theta results
		f=open(results_file, 'a')
		f.write('# Theta koeficienti, ki ustrezajo naslednjim labelom:\n')
		line='; '.join(thetaText)
		f.write('# N; '+line+'\n')
		f.close()

	
	# Racunam za 1. zvezdo na seznamu: to zvezdo izkljucim, natreniram cannonov model in z label_v1.py izracunam njene lable
	for i in range(fluxes.shape[1]):
		theta = train_bayesian_second_order_1PX(fluxes[1:,i], labels[1:], pool=POOL)
	
		if POOL.rank==0:
			line='%d; '%i+'; '.join(['%.6e'%x for x in theta])
			f=open(results_file, 'a')
			f.write(line+'\n')
			f.close()
			print line

	POOL.close()

def train_for_1PX(px=0):
	# V-K, J-K, log R'HK
	labels=np.array([[1, x[2]-x[6], x[4]-x[6], x[10], (x[2]-x[6])*(x[2]-x[6]), (x[2]-x[6])*(x[4]-x[6]), (x[2]-x[6])*x[10], (x[4]-x[6])*(x[4]-x[6]), (x[4]-x[6])*x[10], x[10]*x[10]] for x in data])

	# Subtract the mean values of labels
	l0 = [np.mean(labels[:,i+1]) for i in range(labels.shape[1]-1)]
	for i in range(labels.shape[1]-1):
		labels[:,i+1] -= l0[i]

	POOL = emcee.utils.MPIPool()
	if not POOL.is_master():
		POOL.wait()
		sys.exit(0)

	theta = train_bayesian_second_order_1PX(fluxes[1:,px], labels[1:], pool=POOL, plot=True, niter=2000)

	if POOL.rank==0:
		line='; '.join(['%.6e'%x for x in theta])
		
		print 'l0'
		for t, ll in zip(thetaText[1:-1], l0):
			print t, ll
		
		# Model flux for the first star
		i=0
		f=0
		for t, l in zip(theta, labels[0]):
			l1=l
			if i>0 and i<len(theta)-1:
				l1=l+l0[i-1]
				#~ print l, l1, l0[i-1]
			f+=t*l
			
			#~ print str(t).rjust(10, ' '), str(l).rjust(10, ' '), str(t)*l1.rjust(10, ' '), str(f).rjust(10, ' ')
			
			print '%17s; %.5f; %.5f; %.5f; %.5f; %.5f'%(thetaText[i], l1, l, t, t*l, f)
			#~ print '%17s; %s; %s; %s; %s'%(thetaText[i], str(t).rjust(10, ' '), str(l).rjust(10, ' '), str(t)*l1.rjust(10, ' '), str(f).rjust(10, ' '))
			i+=1
		print 'true f', fluxes[0,px], 'model flux', f, 'delta f', f-fluxes[0,px]

	POOL.close()
	
if __name__ == "__main__":
	#~ train_without_1_star() # Cel spekter
	
	train_for_1PX(px=93) # Samo izbran pixel, 93: 8542
