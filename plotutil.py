
def get_ax(fig, *, nx=1, idx=1, ny=1, title='', xlabel='x', ylabel='', square=False, logaxis=[], projection=None) :
    ax = fig.add_subplot(ny, nx, idx, projection=projection, xlabel=xlabel, ylabel=ylabel, title=title)
    if 'x' in logaxis : ax.set_xscale('log')
    if 'y' in logaxis : ax.set_yscale('log')
    if square :
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.set_aspect('equal', 'box')
    else :
        ax.grid(True)
    return ax
