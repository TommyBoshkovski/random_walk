from numba import cuda, float64
import scipy.io as sci
import numpy as np
import time as tim
import sys
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64


@cuda.jit(device=True)
def get_node(p, cn, rng_state):
    prob = p
    currnode = cn
    rng_states = rng_state
    thread_id = cuda.grid(1)
    r = cuda.random.xoroshiro128p_uniform_float64(rng_states,thread_id)
    upto = 0
    c=0
    while c < 116:
        #print(upto)
        if (upto + prob[(currnode)*116+c]) >= r:
            currnode = c
            break
        upto += prob[(currnode)*116+c]
        c += 1
    return currnode


@cuda.jit(device=True)
def loop(p, t, so, ta, stepp, ste, rng_state, reg):
    source = so -1
    targ = ta -1
    regions = reg
    steps = stepp
    rng_states = rng_state
    prob = p
    time = t
    delay = 0;
    currnode = source
    prevnode = -1;
    step = 0
    if source != targ:
        while  currnode != targ:
            step += 1
            currnode = get_node(prob,currnode, rng_states)
            if prevnode != -1:
		        delay += time[(prevnode)*116+currnode]
            if step>13456:
                step = 13456
                break
            prevnode = currnode
    ste[0]= delay
    return step
    
    

@cuda.jit
def random_walk(prob, time, regions, steps, delay, ste, rng_states):
    targ = cuda.threadIdx.x+1
    source = cuda.blockIdx.x + 1
    p = prob
    t = time
    r=regions
    ids = (source-1)*116 + targ-1
    c = cuda.local.array(shape=4,dtype=float64)
    steps[ids] = loop(p, t, source,targ,steps, c, rng_states,r)
    delay[ids] = c[0]
    #print("source: ", source, " target: ", targ, "  steps:  ", steps[ids], " delay: ")
    cuda.syncthreads() 
    
start_time = tim.time()    
cuda.config.ENABLE_CUDASIM = 1
name = sys.argv[1]
s = sci.loadmat(name+"_prob.mat")
s2 = sci.loadmat(name+"_time.mat")
prob = np.matrix(s[name+"_prob"], dtype=np.float64)
time = np.matrix(s2[name+"_time"], dtype=np.float64)
p = np.array(prob)
p = p.ravel()
t = np.array(time)
t = t.ravel()
regions = np.array(range(1,117), dtype=np.uint8)
cuda.select_device(0)
stream = cuda.stream()
steps = np.zeros((116*116,), dtype=np.uint64,)
ste = np.zeros((116*116,), dtype=np.float64,)
tt = np.zeros((116*116,), dtype=np.uint64,)
stream = cuda.stream()
g_prob = cuda.to_device(p,stream=stream)
g_time = cuda.to_device(t,stream=stream)
g_reg = cuda.to_device(regions,stream=stream)
g_ste = cuda.to_device(ste,stream=stream)
rng_states = create_xoroshiro128p_states(13456, seed=int(tim.time()))
g_steps = cuda.to_device(steps,stream=stream)
g_del = cuda.to_device(tt,stream=stream)
ranwalk = np.empty((10000,116,116), dtype=g_steps.dtype)
delay = np.empty((10000,116,116), dtype=g_steps.dtype)
output = np.empty(shape=g_steps.shape, dtype=g_steps.dtype)
delay_o = np.empty(shape=g_steps.shape, dtype=g_steps.dtype)

for i in range(10000):
   
    random_walk[116, 116](g_prob, g_time, g_reg, g_steps, g_del, g_ste, rng_states)
    print(i)    
    #print("g_steps size:", g_steps.size, " output size: ", output.size)
    cuda.cudadrv.driver.Context.synchronize(cuda.current_context())
    #time.sleep(0.7)
    #print("synchronized")
    output = g_steps.copy_to_host(stream=stream)
    delay_o = g_del.copy_to_host(stream=stream)
    #print("ovde zaglaviv")
    sd = output
    sc = delay_o
    ranwalk[i,:,:] = sd.reshape((116,116))
    delay[i,:,:] = sc.reshape((116,116))
    del rng_states
    #del g_steps
    rng_states = create_xoroshiro128p_states(116*116, seed=np.uint64(tim.time()))
    #g_steps = cuda.to_device(steps)
    
sci.savemat(name +'_randomwalk_steps.mat', {'ranwalk':ranwalk})
sci.savemat(name +'_randomwalk_time.mat', {'delay':delay})
cuda.close()
elapsed_time = tim.time() - start_time
print("elapsed time: ", elapsed_time)
