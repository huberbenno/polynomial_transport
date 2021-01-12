import numpy as np

def assert_close(val1, val2) :
    if hasattr(val1, '__len__') and hasattr(val2, '__len__') :
        if len(val1) != len(val2) :
            print('Values have different lengths: {} and {}!'.format(len(val1), len(val2)))
            assert(False)
        for i in range(len(val1)) :
            assert_close(val1[i], val2[i])
    elif hasattr(val1, '__len__') or hasattr(val2, '__len__') :
        print('One value is a sequence! val1: {}, val2: {}'.format(val1, val2))
        assert(False)
    elif not np.isclose(val1, val2, atol=1e6) :
        print('Got: {} - expected: {}'.format(val1, val2))
        assert(False)
