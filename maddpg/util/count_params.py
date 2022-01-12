"""Computes number of parameters used by PIC in multi-agent env."""
import numpy as np
import argparse

def count(n, h):
    """
    Args:
        n: number of agents
        h: size of hidden layer used by PIC
    """
    # self pos, self vel
    # all landmark pos
    # all other agents pos
    if n == 3:
        dim_obs = 4 + n*2 + (n-1)*2
    else:
        # They let agents observe only the 5 nearest agents
        # See Appendix B of PIC paper
        dim_obs = 26  
    dim_action = 5
    
    # s = n * dim_obs
    # a = n * dim_action
    s = dim_obs
    a = dim_action
    # PIC is (|S|+|A|)*H*2 + H*H*2 + H
    count_pic = (s + a)*h*2 + h*h*2 + h

    print('PIC has', count_pic, ' parameters.')

    # DeepSet is (|S|+|A|)*H + H*H + H
    coeff = [1, (s+a+1), -count_pic]
    count_deepset = np.max(np.roots(coeff))
    
    print('DeepSet needs hidden layer size', count_deepset)

    # MLP has (|S|+|A|)*N*H + H*H + H
    coeff = [1, (n*(s+a)+1), -count_pic]
    count_mlp = np.max(np.roots(coeff))
    print('MLP needs hidden layer size', count_mlp)

    # MF-Q has (|S|+2*|A|)*H + H*H + H
    coeff = [1, ((s + 2*a)+1), -count_pic]
    count_mf = np.max(np.roots(coeff))
    print('Mean field Q needs hidden layer size', count_mf)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('h', type=int)
    args = parser.parse_args()

    count(args.n, args.h)
