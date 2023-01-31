import numpy as np

def close(val1, val2, atol=1e-6) :
    if hasattr(val1, '__len__') and hasattr(val2, '__len__') :
        if len(val1) != len(val2) :
            print('Values have different lengths: {} and {}!'.format(len(val1), len(val2)))
            assert(False)
        for i in range(len(val1)) :
            close(val1[i], val2[i])
    elif hasattr(val1, '__len__') or hasattr(val2, '__len__') :
        print('One value is a sequence! val1: {}, val2: {}'.format(val1, val2))
        assert(False)
    elif not np.isclose(val1, val2, atol=atol) :
        print('Got: {} - expected: {}'.format(val1, val2))
        assert(False)

def equal(val1, descr1, val2, descr2) :
    if val1 != val2 :
        print('{} ({}) != {} ({})'.format(val1, descr1, val2, descr2))
        assert(False)