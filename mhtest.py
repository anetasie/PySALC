# metropolis multivariate normal fit
# cash statistic is -2*loglikelihood
# fitted value for mean and covariances for multivariate normal from sherpa

import numpy as np
from sherpa.astro.ui import *
import sherpa.astro.ui as ui
#import scipy.special as sp
from funcs import *
from sherpa.models.parameter import ParameterErr


def dmvnorm(x, mu, sigma, log=True):
	if np.min( np.linalg.eigvalsh(sigma))<=0 :
		raise RuntimeError, "Error: sigma is not positive definite"
	if np.max( np.abs(sigma-sigma.T))>=1e-9 :
		raise RuntimeError, "Error: sigma is not symmetric"
		
	logdens = -mu.size/2.0*log(2*pi)- 1/2.0*np.log( np.linalg.det(sigma) )-1/2.0 * np.dot( x-mu, np.dot(np.linalg.inv(sigma), x-mu ) )
	if log:
		return logdens
	else:
		dens = exp( logdens )
		return dens

def factorial( x, log=True):
	if type(x) != int:
		raise RuntimeError, "Must be an integer"
	if x==0 or x==1:
		if log:
			return 0
		else:
			return 1
			
	if x > 1:
		y = range(1,x+1)
		if log:
			result = np.sum( np.log(y))
			return result
		else:
			result = np.prod( y )
			return result

gammln_cof = np.array([76.18009173, -86.50532033, 24.01409822,
	-1.231739516e0, 0.120858003e-2, -0.536382e-5])
gammln_stp = 2.50662827465

#============= Gamma, Incomplete Gamma ===========

def gammln(xx):
	"""Logarithm of the gamma function."""
	if xx==1 or xx==2:
		return 0
	else:
		global gammln_cof, gammln_stp
		x = xx - 1.
		tmp = x + 5.5
		tmp = (x + 0.5)*np.log(tmp) - tmp
		ser = 1.
		for j in range(6):
			x = x + 1.
			ser = ser + gammln_cof[j]/x
		return tmp + np.log(gammln_stp*ser)


def dmvt( x, mu, sigma, df, log=True, norm=False):
	
	if np.min( np.linalg.eigvalsh(sigma))<=0 :
		raise RuntimeError, "Error: sigma is not positive definite"
	if np.max( np.abs(sigma-sigma.T))>=1e-9 :
		raise RuntimeError, "Error: sigma is not symmetric"
		
	p = mu.size
	logdens_unnorm = -.5*np.log( np.linalg.det(sigma) ) - (df+p)/2.0*np.log( df + np.dot( x-mu, np.dot(np.linalg.inv(sigma), x-mu ) ) )
	if log :
		if norm:
			logdens = logdens_unnorm + gammln( (df+p)/2. )-gammln(df/2.)-(p/2.)*np.log(np.pi)+(df/2.)*np.log(df)
			return logdens
		else:
			return logdens_unnorm
		
	else:
		if norm:
			logdens = logdens_unnorm + gammln( (df+p)/2. )-gammln(df/2.)-(p/2.)*np.log(np.pi)+(df/2.)*np.log(df)
			dens = np.exp( logdens )
			return dens
		else:
			dens_unnorm = np.exp( logdens_unnorm )
			return dens_unnorm

def _set_par_vals(parnames, parvals):
	"Sets the paramaters to the given values"

	for (parname,parval) in zip(parnames,parvals):
		ui.set_par(parname, parval)


def mht(parnames, mu, sigma, num_iter, df, MH=True, multmodes=False, log=False, inv=False, defaultprior=True, priorshape=False, originalscale=True, verbose=False, scale=1, sigma_m=False, p_M=.5, maxconsrej=100, savedraws=True, thin=1):
	"""
	p_M is mixing proportion of Metropolis draws in the mixture of MH and Metropolis draws
	"""
	prior=np.repeat( 1.0, parnames.size)
	priorshape = np.array(priorshape)
	originalscale = np.array(originalscale)
	# if not default prior, prior calculated at each iteration
	if defaultprior!=True:
		if priorshape.size!=parnames.size:
			raise RuntimeError, "If not using default prior, must specify a function for the prior on each parameter"
		if originalscale.size!=parnames.size:
			raise RuntimeError, "If not using default prior, must specify the scale on which the prior is defined for each parameter"
	
	jacobian = np.repeat( False, parnames.size)
	### jacobian needed if transforming parameter but prior for parameter on original scale
	if defaultprior!=True:
		### if log transformed but prior on original scale, jacobian for those parameters is needed
		if np.sum( log*originalscale ) > 0:
			jacobian[ log*originalscale ] = True
		if np.sum( inv*originalscale ) > 0:
			jacobian[ inv*originalscale ] = True
	
	log = np.array(log)
	if log.size==1:
		log = np.tile( log, parnames.size)
		
	inv = np.array(inv)
	if inv.size==1:
		inv = np.tile( inv, parnames.size)
	if MH:
		print "Running Metropolis-Hastings"
	else:
		print "Running Metropolis and Metropolis-Hastings"
	if multmodes:
		print "Will add second mode if gets stuck"
	print "\nPrior"

	iterations = np.zeros( (num_iter+1, mu.size) )
	statistics = np.zeros( (num_iter+1,1) )
	
	current=np.copy(mu)
	_set_par_vals(parnames, current)
	
	statistics[0] = -.5*calc_stat()
	if defaultprior!=True:
		x=np.copy(current)
		if np.sum(originalscale) < parnames.size:
			for i in range(parnames.size):
				if log[i]*(1-originalscale[i])>0:
					x[i]=np.log( x[i])
				if inv[i]*(1-originalscale[i])>0:
					x[i]=1.0/x[i]
			for par in range(0, parnames.size):
				prior[par] = eval_prior( x[par], priorshape[par])

	statistics[0] += np.sum( np.log( prior) )
	if np.sum(log*jacobian)>0:					
		statistics[0] += np.sum( np.log( current[log*jacobian] ) )
	if np.sum(inv*jacobian)>0:
		statistics[0] += np.sum( 2.0*np.log( np.abs(current[inv*jacobian]) ) )

	# using delta method to create proposal distribution on log scale for selected parameters
	if np.sum(log)>0:
		logcovar = np.copy(sigma)
		logcovar[:,log]= logcovar[:,log]/mu[log]
		logcovar[log]= (logcovar[log].T/mu[log]).T
		sigma = np.copy(logcovar)
		mu[log]=np.log(mu[log])
		current[log]=np.log( current[log])
	
	# using delta method to create proposal distribution on inverse scale for selected parameters
	if np.sum(inv)>0:
		invcovar = np.copy(sigma)
		invcovar[:,inv]= invcovar[:,inv]/(-1.0*np.power(mu[inv],2))
		invcovar[inv]= (invcovar[inv].T/(-1.0*np.power(mu[inv],2))).T
		sigma = np.copy(invcovar)
		mu[inv]=1.0/(mu[inv])
		current[inv]=1.0/( current[inv])
		
	iterations[0] = np.copy(mu) 
	zero_vec = np.zeros(mu.size)
	rejections=0
	modes = 1
	if np.mean(sigma_m) == False:
		sigma_m=np.copy(sigma)
		
	for i in range(1,num_iter+1,1):
		current = iterations[i-1]
		if verbose:
			if np.mod(i,1000)==0:
				print "draw "+str(i)
		q = np.random.chisquare(df, 1)[0]
		if MH:
			while True:
				try:
					if modes==1 :
						proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
					else:
						u = np.random.uniform(0,1,1)
						if u <= p :
							proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
						else:
							proposal = mu2 + np.random.multivariate_normal(zero_vec, sigma2)/ np.sqrt(q/df)
					if np.sum(log)>0:
						proposal[log]=np.exp(proposal[log])
					if np.sum(inv)>0:
						proposal[inv]=1.0/proposal[inv]
						
					_set_par_vals(parnames, proposal)
					if defaultprior!=True:
						x=np.copy(proposal)
						### is prior for all parameters evaluated on the original scale?
						if np.sum(originalscale) < parnames.size:
							for i in range(parnames.size):
								if log[i]*(1-originalscale[i])>0:
									x[i]=np.log( x[i])
								if inv[i]*(1-originalscale[i])>0:
									x[i]=1.0/x[i]
						for par in range(0, parnames.size):
							prior[par] = eval_prior( x[par], priorshape[par])
						
					#putting parameters back on log scale
					if np.sum(log)>0:
						proposal[log] = np.log( proposal[log] )
					#putting parameters back on inverse scale
					if np.sum(inv)>0:
						proposal[inv] = 1.0/proposal[inv]
						
					break
				except ParameterError:
					pass
			
			stat_temp = -.5*calc_stat()

			stat_temp += np.sum( np.log( prior))
			# adding jacobian (if necessary) with parameters on the log scale sum( log(theta)), but everything stored on log scale
			if np.sum(log*jacobian)>0:					
				stat_temp += np.sum( proposal[log*jacobian] )
			# adding jacobian (if necessary) with parameters on the inverse scale, sum(2*log(theta))=-sum(2*log(phi)), 
			if np.sum(inv*jacobian)>0:
				stat_temp -= np.sum( 2.0*np.log( np.abs(proposal[inv*jacobian]) ) )
		
			if modes==1 :
				alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )
			else:
				alpha = np.exp( stat_temp + evalmixture(current, mu, mu2, sigma, sigma2, df,p) - statistics[i-1] - evalmixture(proposal, mu, mu2, sigma, sigma2, df, p) )
			
					
		else:
			u = np.random.uniform(0,1,1)
			#if np.mod(i,2)==1:
			if u <= p_M:
				#Metropolis jumping rule
				while True:
					try:
						#if ((sigma==sigma_m).all()):
						proposal = iterations[i-1] + np.random.multivariate_normal(zero_vec, sigma_m*scale)/ np.sqrt(q/df)
						if np.sum(log)>0:
							proposal[log]=np.exp(proposal[log])
						if np.sum(inv)>0:
							proposal[inv]=1.0/proposal[inv]
	
						#else:
							#if np.sum(log)>0:
								#temp_iter = np.copy( iterations[i-1])
								#temp_iter[log] = np.exp( temp_iter[log])
								#proposal = temp_iter + np.random.multivariate_normal(zero_vec, sigma_m*scale)/ np.sqrt(q/df)
						_set_par_vals(parnames, proposal)
						if defaultprior!=True:
							x=np.copy(proposal)
							### is prior for all parameters evaluated on original scale?
							if np.sum(originalscale) < parnames.size:
								for i in range(parnames.size):
									if log[i]*(1-originalscale[i])>0:
										x[i]=np.log( x[i])
									if inv[i]*(1-originalscale[i])>0:
										x[i]=1.0/x[i]
							for par in range(0, parnames.size):
								prior[par] = eval_prior( x[par], priorshape[par])
						if np.sum(log)>0:
							proposal[log] = np.log( proposal[log] )
						if np.sum(inv)>0:
							proposal[inv]=1.0/proposal[inv]
								
						break
					except ParameterError:
						pass
				stat_temp = -.5*calc_stat()
				stat_temp += np.sum( np.log(prior))
				# adding jacobian (if necessary) with parameters on the log scale sum( log(theta)), but everything stored on log scale
				if np.sum(log*jacobian)>0:					
					stat_temp += np.sum( proposal[log*jacobian] )
				# adding jacobian (if necessary) with parameters on the inverse scale, sum(2*log(theta))=-sum(2*log(phi)), 
				if np.sum(inv*jacobian)>0:
					stat_temp -= np.sum( 2.0*np.log( np.abs(proposal[inv*jacobian]) ) )

				alpha = np.exp( stat_temp - statistics[i-1])

			else:
			#if np.mod(i,2)==0:
				#MH jumping rule
				while True:
					try:
						if modes==1 :
							proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
						else:
							u = np.random.uniform(0,1,1)
							if u <= p :
								proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
							else:
								proposal = mu2 + np.random.multivariate_normal(zero_vec, sigma2)/ np.sqrt(q/df)
						# back transform proposal for evaluation on original scale
						if np.sum(log)>0:
							proposal[log]=np.exp(proposal[log])
						if np.sum(inv)>0:
							proposal[inv]=1.0/proposal[inv]
							
						_set_par_vals(parnames, proposal)
						if defaultprior!=True:
							x=np.copy(proposal)
							### is prior for all parameters evaluated on original scale
							if np.sum(originalscale) < parnames.size:
								for i in range(parnames.size):
									if log[i]*(1-originalscale[i])>0:
										x[i]=np.log( x[i])
									if inv[i]*(1-originalscale[i])>0:
										x[i]=1.0/x[i]
							for par in range(0, parnames.size):
								prior[par] = eval_prior( x[par], priorshape[par])
						# transform parameter
						if np.sum(log)>0:
							proposal[log] = np.log( proposal[log] )
						if np.sum(inv)>0:
							proposal[inv] = 1.0/proposal[inv]
						
						break
					except ParameterError:
						pass
				stat_temp = -.5*calc_stat()
				stat_temp += np.sum( np.log( prior))
				# adding jacobian (if necessary) with parameters on the log scale sum( log(theta)), but everything stored on log scale
				if np.sum(log*jacobian)>0:					
					stat_temp += np.sum( proposal[log*jacobian] )
				# adding jacobian (if necessary) with parameters on the inverse scale, sum(2*log(theta))=-sum(2*log(phi)), 
				if np.sum(inv*jacobian)>0:
					stat_temp -= np.sum( 2.0*np.log( np.abs(proposal[inv*jacobian]) ) )
				
				if modes==1 :
					alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )

				else:
					alpha = np.exp( stat_temp + evalmixture(current, mu, mu2, sigma, sigma2, df,p) - statistics[i-1] - evalmixture(proposal, mu, mu2, sigma, sigma2, df, p) )
								
		u = np.random.uniform(0,1,1)
		if u <= alpha:
			iterations[i]=np.copy(proposal)
			
			statistics[i]=np.copy(stat_temp)
			rejections=0
		else:
			
			iterations[i]=np.copy(iterations[i-1])
			statistics[i]=np.copy(statistics[i-1])
			
			### added for test
			rejections += 1
			if ( rejections > maxconsrej and modes==1 and multmodes ):
				print "need a second mode"
				modes=2
				mu2 = iterations[i]
				print mu2
				if sigma.size > 1:
					eigenvals = np.linalg.eigvalsh(sigma)
				else:
					eigenvals = sigma
				sigma2 = np.diag( np.repeat(max(eigenvals),mu.size) )
				#sigma2 = sigma
				d1 = np.exp(-.5*statistics[0])
				d2 = np.exp(-.5*statistics[i])
				
				### returning normalized densities (not log)
				t11 = dmvt( mu, mu, sigma, df, False, True)
				t21 = dmvt( mu, mu2, sigma2, df, False, True)
				t12 = dmvt( mu2, mu, sigma, df, False, True)
				t22 = dmvt( mu2, mu2, sigma2, df, False, True)
				
				
				### p calculated with sigma2
				#a = np.array([[d1[0],t21-t11],[d2[0],t22-t12]])
				#b = np.array([t21,t22])
				#x = np.linalg.solve( a, b)
				#p = x[1]
				
				p = d1/(d1+d2)
				papprox = 1/ ( np.exp(np.log(d2)-np.log(d1)+np.log(t11)-np.log(t22)) +1)
				
				print "p is" 
				print p
				print "p approx is"
				print papprox
				
	if np.sum(log)>0:
		iterations[:,log] = np.exp( iterations[:,log])
	if np.sum(inv)>0:
		iterations[:,inv] = 1.0/( iterations[:,inv])
		
	result = np.hstack( (statistics,iterations) )
	return result





def mhttest(mu,sigma,num_iter, df, ptruth, dist, multmodes=True):
	iterations = np.zeros( (num_iter+1, mu.size) )
	statistics = np.zeros( (num_iter+1,1) )
	iterations[0] = mu
	current=mu
	
	### added for test
	mu2truth = mu + np.dot( np.repeat(dist,mu.size) , sigma )
	if sigma.size > 1:
		eigenvals = np.linalg.eigvalsh(sigma)
	else:
		eigenvals = sigma
	sigma2truth = np.diag( np.repeat(max(eigenvals),mu.size) )
	
	###set_par("abs1.nh",current[0])
	###set_par("p1.gamma",current[1])
	###set_par("p1.ampl",current[2])
	
	#statistics[0] = calc_stat()
	
	### added for test
	statistics[0] = evalmixture(current, mu, mu2truth, sigma, sigma2truth, df, ptruth)

	zero_vec = np.zeros(mu.size)
	
	### added for test
	rejections=0
	modes = 1
	
	for i in range(1,num_iter+1,1):
		current = iterations[i-1]
		q = np.random.chisquare(df, 1)[0]
		if modes==1 :
			proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
		### added for test
		else:
			u = np.random.uniform(0,1,1)
			if u <= p :
				proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
			else:
				proposal = mu2 + np.random.multivariate_normal(zero_vec, sigma2)/ np.sqrt(q/df)
				
		###set_par("abs1.nh",proposal[0])
		###set_par("p1.gamma",proposal[1])
		###set_par("p1.ampl",proposal[2])
		###stat_temp = calc_stat()
		
		### added for test
		stat_temp = evalmixture( proposal, mu, mu2truth, sigma, sigma2truth, df, ptruth)
		
		if modes==1 :
			#alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )
			alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )

		else:
			#alpha = np.exp( stat_temp + p*dmvt(current, mu, sigma, df, True, True)+(1-p)*dmvt(current, mu2, sigma2, df, True, True) - statistics[i-1] - p*dmvt(proposal, mu, sigma, df, True, True)-(1-p)*dmvt(proposal, mu2, sigma2, df, True, True) )
			alpha = np.exp( stat_temp + evalmixture(current, mu, mu2, sigma, sigma2, df,p) - statistics[i-1] - evalmixture(proposal, mu, mu2, sigma, sigma2, df, p) )
			
		u = np.random.uniform(0,1,1)
		if u <= alpha:
			iterations[i]=proposal
			statistics[i]=stat_temp
			rejections=0
		else:
			
			iterations[i]=iterations[i-1]
			statistics[i]=statistics[i-1]
			
			### added for test
			rejections += 1
			if ( rejections >100 and modes==1 and multmodes ):
				print "need a second mode"
				modes=2
				mu2 = iterations[i]
				print mu2
				if sigma.size > 1:
					eigenvals = np.linalg.eigvalsh(sigma)
				else:
					eigenvals = sigma
				sigma2 = np.diag( np.repeat(max(eigenvals),mu.size) )
				d1 = np.exp(statistics[0])
				d2 = np.exp(statistics[i])
				t11 = dmvt( mu, mu, sigma, df, False, True)
				t21 = dmvt( mu, mu2, sigma2, df, False, True)
				t12 = dmvt( mu2, mu, sigma, df, False, True)
				t22 = dmvt( mu2, mu2, sigma2, df, False, True)
				a = np.array( [[t11,t21],[t12,t22]] )
				b = np.array( [d1,d2] )
				#assumed posterior not normalized, thus solving for pvec/k
				x = np.linalg.solve( a, b)
				# normalize to find p
				p = x[0]/np.sum(x)
				
				### alternative calculation
				
				a = np.array([[d1[0],t21-t11],[d2[0],t22-t12]])
				b = np.array([t21,t22])
				x = np.linalg.solve( a, b)
				p2 = x[1]
				
				papprox = 1/ ( np.exp(np.log(d2)-np.log(d1)+np.log(t11)-np.log(t22)) +1)
				print "p is" 
				print p
				print "p constrained between 0 and 1 is "
				print p2
				print "p approx is"
				print papprox
				
				
	result = np.hstack( (statistics,iterations) )
	return result

def comp_prop(statistics, max_stat, cutoffs):
	ncutoffs = len(cutoffs)
	prop = np.zeros(ncutoffs)
	for i in range(ncutoffs):
		prop[i] = mean( statistics < max_stat+cutoffs[i])
	return prop
		
	
def evalmixture( current, mu1, mu2, sigma1, sigma2, df, p, log=True ):
	post = p*dmvt( current, mu1, sigma1, df, False, True ) + (1-p)*dmvt( current, mu2, sigma2, df, False, True)
	if log:
		logpost = np.log( post )
		return logpost
	return post


### test 1
### suppose the posterior distribution were exactly a mixture of two multivariate t's
### further, suppose that we know the mean of the major mode is mu and sigma 
def write_draws(draws, outfile):
	"Writes the draws to the file called outfile"
	fout = open(outfile, "w")
	for line in draws:
		for element in line:
			fout.write("%s " % element)
		fout.write("\n")
	fout.close()
	
#mu = np.array([0,0])
#sigma = np.array([[1,0],[0,2]])
#result = mhttest( mu, sigma, 10000, 4 , .9, 8,True)
#write_draws(result, "C:\Users\Jason\outfile")
