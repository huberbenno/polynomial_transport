def print_start(msg, end=' ') :
    print('+-------------------------------------------------+')
    print('| ' + msg, end=end)
    if end == '\n' :
        print('+ - - - - - - - - - - - - - - - - - - - - - - - - +')

def print_done_ctd(msg='Done.', end='\n') :
    print(' ' + msg, end=end)
    print('+-------------------------------------------------+')

def print_done() :
    print('+ - - - - - - - - - - - - - - - - - - - - - - - - +')
    print('|  Done. ')
    print('+-------------------------------------------------+')

def print_indent(msg=' ', level=1) :
    print('| ' + '\t' * level + msg)