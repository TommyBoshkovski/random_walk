from numba import cuda
import scipy.io as sci
import numpy as np
import time
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
    c=1
    while c < 117:
        #print(upto)
        if (upto + prob[(currnode-1)*116+c-1]) >= r:
            currnode = c
            break
        upto += prob[(currnode-1)*116+c-1]
        c += 1
    return currnode


@cuda.jit(device=True)
def loop(p, so, ta, stepp, rng_state, reg):
    source = so
    targ = ta
    regions = reg
    steps = stepp
    rng_states = rng_state
    prob = p
    currnode = source
    step = 0
    if source != targ:
        while  currnode != targ:
            step += 1
            currnode = get_node(prob,currnode, rng_states)
            if step>13456:
                step = 13456
                break
    return step

@cuda.jit
def random_walk(prob, regions, steps, rng_states):
    targ = cuda.threadIdx.x+1
    source = cuda.blockIdx.x + 1
    p = prob
    r=regions
    ids = (source-1)*116 + targ-1
    steps[ids] = loop(p, source,targ,steps,rng_states,r)
    #print("source: ", source, " target: ", targ, "  steps:  ", steps[ids])
    cuda.syncthreads() 
    
start_time = time.time()    
cuda.config.ENABLE_CUDASIM = 1
name = sys.argv[1]
s = sci.loadmat(name)
prob = np.matrix(s[name[:-4]], dtype=np.float64)
p = np.array(prob)
p = p.ravel()
regions = np.array(range(1,117), dtype=np.uint8)
cuda.select_device(1)
stream = cuda.stream()
steps = np.zeros((116*116,), dtype=np.uint64,)
stream = cuda.stream()
g_prob = cuda.to_device(p,stream=stream)
g_reg = cuda.to_device(regions,stream=stream)
rng_states = create_xoroshiro128p_states(13456, seed=int(time.time()))
g_steps = cuda.to_device(steps,stream=stream)
ranwalk = np.empty((10000,116,116), dtype=g_steps.dtype)
output = np.empty(shape=g_steps.shape, dtype=g_steps.dtype)

for i in range(10000):
   
    random_walk[116, 116](g_prob, g_reg, g_steps,rng_states)
    print(i)    
    #print("g_steps size:", g_steps.size, " output size: ", output.size)
    cuda.cudadrv.driver.Context.synchronize(cuda.current_context())
    #time.sleep(0.7)
    #print("synchronized")
    output = g_steps.copy_to_host(stream=stream)
    #print("ovde zaglaviv")
    sd = output;
    ranwalk[i,:,:] = sd.reshape((116,116))
    del rng_states
    #del g_steps
    rng_states = create_xoroshiro128p_states(116*116, seed=np.uint64(time.time()))
    #g_steps = cuda.to_device(steps)

    
sci.savemat(name[:-4] +'_randomwalk.mat', {'ranwalk':ranwalk})
cuda.close()
elapsed_time = time.time() - start_time
print("elapsed time: ", elapsed_time)
