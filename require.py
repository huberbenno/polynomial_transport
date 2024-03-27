import numpy as np

def cpstr(val1, val2, descr1, descr2) :
    return '\n\t\t {} : {} \n\t\t {} : {}'.format(val1, descr1, val2, descr2)

def check_collection(check_single, val1, val2, descr1, descr2) :
    if hasattr(val1, '__len__') and hasattr(val2, '__len__') :
        if len(val1) != len(val2) :
            print('Values have different lengths!' + cpstr(val1, val2, descr1, descr2))
            assert(False)
        for i in range(len(val1)) :
            check_collection(check_single, val1[i], val2[i],
                             str(i) + ' ' + descr1, str(i) + ' ' + descr2)
    elif hasattr(val1, '__len__') or hasattr(val2, '__len__') :
        print('One value is a sequence!' + cpstr(val1, val2, descr1, descr2))
        assert(False)
    else :
        check_single(val1, val2, descr1, descr2)

def close_single(val1, val2, descr1, descr2, atol=1e-6) :
    assert(not hasattr(val1, '__len__'))
    assert(not hasattr(val2, '__len__'))
    if not np.isclose(val1, val2, atol=atol) :
        print('Values not close (tol = {})!'.format(atol) + cpstr(val1, val2, descr1, descr2))
        assert(False)

def close(val1, val2, descr1='...', descr2='...', atol=1e-6) :
    single = lambda val1, val2, descr1, descr2 : close_single(val1, val2, descr1, descr2, atol=atol)
    check_collection(single, val1, val2, descr1, descr2)

def equal_single(val1, val2, descr1, descr2) :
    if val1 != val2 :
        print('Values not equal!' + cpstr(val1, val2, descr1, descr2))
        assert(False)

def equal(val1, val2, descr1='...', descr2='...') :
    check_collection(equal_single, val1, val2, descr1, descr2)

def notNone(val, descr) :
    if val is None :
        print(descr + 'is None!')
        assert(False)