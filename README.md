# RegNet
Implementation of RegNet (In Tensorflow)

Reference Paper: "Designing Network Design Spaces"

Below is the short summary of the reference paper:

- - - - - - - - 


- Network:
 - stem
 - body 
 - head
 
- body:
 - stage1
 - stage2
 - stage3
 - stage4
 
- stagei
 - block1
 - block2
 - block3
 - ...
 - blockdi
 
- stem
 - 3x3 Conv stride=2 filters=w0 (32)
 
- head 
 - AvgPool
 - Dense units=n (for n classes)

- stage parameter
 - number of blocks **di**

- block parameters
 - width **wi**
 - bottleneck ration **bi**
 - group width **gi**
 

- All the blocks are identical except the first block
- The first block uses stride=2 Conv.
- wi refers to number of channels (in a block)
- r,r refers resolution/ width and height of feature map outputs
- body contains only 4 stages  

- - - - -- 

- AnyNetXa:
 - any possible model within its parameters combinations
 
- AnyNetXb:
 - bottleneck ratio **bi** is fixed across all stages
 
- AnyNetXc:
 - group width **gi** is fixed across all stages
 
- AnyNetXd:
 - stage width wi+1 is greater than previous width wi
 
- AnyNetXe:
 - stage depth di+1 is greater than previous depth di
 
- RegNet:
 - per block width wj, where j is index of blocks. 
 - observations:
   - found that good models in design space have linear fit for block width wj with their position j
   - wj = 48*(j+1) for 0<=j<=20
 - Proposed approach:
   - d total depth, j index of block position
   - uj = w0 + wa*j  for 0<=j<d (Eqn1)
   - w0 is initial width (>0)
   - wa slope (>0)
   - we introduce another additional parameter wm (>0)
   - given uj, wm now find value of sj such that it satisfies the following eqn
   - uj = w0* (wm)**sj (Eq2)
   - compute sj for each block j
   - to quantize wj we round off sj
   - i.e. [sj] (rounded off)
   - Now we compute per block width wj by
   - wj = w0* (wm)**[sj] (Eqn3)
  - 6 parameters:
    - d, w0, wa, wm, b, g
  - Sampled models have constraints:
    - d < 64
    - w0, wa < 256
    - 1.5 <= wm <= 3
    - b <= 2
    - g > 1
   - good model observed parameters:
     - wm =2 
     - w0 = wa
     - observation that the third stage has higher number of blocks whereas the last stage has smaller number of blocks.
     - g increases with more large models, whereas the d saturates for large models.
 - RegNetX-200MF
     - di = [1,1,4,7]
     - wi = [24,56,152,368]
     - g = 8
     - b = 1
     - wa = 36, w0 = 24, wm =2.5
     - 2.7 Million Parameters
     - error rate 30.8%
 - RegNetX-400MF
     - di = [1,2,7,12]
     - wi = [32,64,160,384]
     - g = 16
     - b = 1
     - wa = 24, w0 = 24, wm =2.5
     - 5.2 Million Parameters
     - error rate 27.2%
 - RegNetX-600MF
     - di = [1,3,5,7]
     - wi = [48,96,240,528]
     - g = 24
     - b = 1
     - wa = 37, w0 = 48, wm =2.2
     - 6.2 Million Parameters
     - error rate 25.5%
 - RegNetX-800MF
     - di = [1,3,7,5]
     - wi = [64,128,288,672]
     - g = 16
     - b = 1
     - wa = 36, w0 = 56, wm =2.3
     - 7.3 Million Parameters
     - error rate 24.8%

- - - - - - - - 
