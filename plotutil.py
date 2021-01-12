
def get_ax(fig, nx, idx, *, ny=1, title='', xlabel='x', ylabel='', logaxis=[], projection=None) :
    ax = fig.add_subplot(ny, nx, idx, projection=projection)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    if 'x' in logaxis : ax.set_xscale('log')
    if 'y' in logaxis : ax.set_yscale('log')
    return ax
