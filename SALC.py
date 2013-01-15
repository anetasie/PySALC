from ui import *
import sherpa.astro.ui as ui
from mhtest import *
execfile("funcs.py")

set_stat("cash")

model = "xsphabs.abs1*powlaw1d.p1"
parnames=np.array(['abs1.nh','p1.gamma','p1.ampl'])

arf1=unpack_arf("core1.arf")
rmf1=unpack_rmf("core1.rmf")

set_model(1, model)

abs1.nh = 0.1
p1.gamma = 2.0

# define the number of counts in the simulation and exposure time

expos=100
counts0=300

fake_pha(1,arf1,rmf1,exposure=expos)
p1.ampl = counts0/calc_data_sum()
fake_pha(1,arf1,rmf1,exposure=expos)

ranges= np.array([[.3,.9],[.9,2.5],[2.5,8.0]])

counts = np.zeros( ranges.shape[0] )
lamb = np.zeros( ranges.shape[0] )
lklhd = np.zeros( ranges.shape[0] )
for i in range(ranges.shape[0]):
	notice( ranges[i,0] , ranges[i,1])
	counts[i] = calc_data_sum(ranges[i,0] , ranges[i,1] )
	lamb[i] = calc_model_sum(ranges[i,0] , ranges[i,1] )
	lklhd[i] = np.exp(-.5*calc_stat() )
	notice()

def log_sum( counts, ranges):
	for i in range(ranges.shape[0]):
		notice( ranges[i,0] , ranges[i,1])
		lamb[i] = calc_model_sum(ranges[i,0] , ranges[i,1] )
		lklhd[i] = counts[i] * np.log(lamb[i]) - lamb[i]
		notice()
	return np.sum(lklhd)

def z(counts,ranges,outfile='z.txt'):
        L   = 30
	out = np.zeros( (L,L) )
        for i in range(1,L,1):
		for j in range(1,L,1):
			if j==1:
				print str(i)
			abs1.nh  = i*0.0333+0.001
			p1.gamma = j*0.1
			out[i,j] = log_sum(counts, ranges)
	write_draws(out, outfile)

def grid(ranges,outfile='grid.txt'):
	n = np.array([0.001,0.01,0.025,0.05,0.075,0.1,0.125,0.25,0.5,1])
	g = np.array([0.1,0.25,0.5,0.75,1,1.5,2,3])
	out  = np.zeros( (g.size*n.size,4) )
	lamb = np.zeros( ranges.shape[0] )
	for i in range(g.size):
		for j in range(n.size):
		 	if j==1:
		       		print str(i)
			abs1.nh  = n[j]
			p1.gamma = g[i]
			for k in range(ranges.shape[0]):
				lamb[k] = calc_model_sum(ranges[k,0], ranges[k,1])
			cS = np.log10(lamb[0]/lamb[1])
			cH = np.log10(lamb[1]/lamb[2])
			out[(n.size)*i+j] = np.hstack( (g[i],n[j],cS,cH) ) 
	write_draws(out, outfile)
		

# Setup for running metropolis, initial parameters and scales for mvn based on covariance
# Note covar - may fail, change to the diagonal if this happens

#start = np.array([.1,.1,.1])
k = ranges.shape[0]-1
notice(ranges[0,0],ranges[k,1])
fit()
covar()

#start = np.array([0.1,1.5,0.0001])  # initial parameters in original code set by hand
start = get_fit_results().parvals

#sigma = np.diag( np.array([.0001,.0001,.0001]) ) # fixed scales in original
#sigma = np.diag(np.array(get_covar_results().parmaxes)**2)  # diagonal matrix, no correlation
sigma = get_covar_results().extra_output


def metropolis(counts, ranges, parnames, start, sigma, outfile='out.txt', num_iter=1000 ):
		
	iterations = np.zeros( (num_iter+1, parnames.size) )
	statistics = np.zeros( (num_iter+1,1) )
	
	current=np.copy(start)
	_set_par_vals(parnames, current)
	statistics[0] = log_sum( counts, ranges)
	statistics[0] += -0.01*current[1] - np.log(current[2])
	
	iterations[0] = np.copy(current) 
	zero_vec = np.zeros(parnames.size)
	for i in range(1,num_iter+1,1):
		if np.mod(i,100)==1:
			print "draw "+str(i)
			#print "current", current
		current = np.copy(iterations[i-1])
		while True:
			try:
				#if ((sigma==sigma_m).all()):
				proposal = iterations[i-1] + np.random.multivariate_normal(zero_vec, sigma)
				_set_par_vals(parnames, proposal)
				break
			except Exception:	
				pass
		stat_temp = log_sum( counts, ranges)
		stat_temp += -0.01*proposal[1] - np.log(proposal[2])
		alpha = np.exp( stat_temp - statistics[i-1])
		u = np.random.uniform(0,1,1)
		if u <= alpha:
			iterations[i]=np.copy(proposal)
			statistics[i]=np.copy(stat_temp)
		else:
			iterations[i]=np.copy(iterations[i-1])
			statistics[i]=np.copy(statistics[i-1])

	burnin = int(np.round(num_iter*0.2)+1)
        result = np.hstack( (statistics[np.arange(burnin,num_iter+1)],iterations[np.arange(burnin,num_iter+1)]) )

	analyze_draws( result, parnames, 2, dict=False, verbose=True, means=True)
	write_draws( iterations, outfile)
	return result
	
draws = metropolis( counts, ranges, parnames, start, sigma, outfile="out.txt", num_iter=10000 )


#means ={}
#for i,par in enumerate(parnames):
#	means[par] = np.round(np.mean( draws[:,i+1]) , 3)

#means
