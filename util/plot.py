import numpy as np
import matplotlib.pyplot as plt


def get_ax(*, fig=None, nx=1, ny=1, idx=1, title='', xlabel='', ylabel='', square=False, logaxis=[], projection=None) :
    if fig is None : fig = plt.figure()
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


def plot_density(*, density=None, grid=None, samples=None, ax=None, fig=None, figsize=12,
                    n=200, xlim=[-1,1], ylim=[-1,1], nlevels=7,
                    alpha=1, qcs=None, cmap='Blues', filename=None) :
    if ax is None :
        if fig is None : fig = plt.figure(figsize=(figsize,figsize))
        ax = get_ax(fig=fig, square=True)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    # density
    x = np.linspace(*xlim,n)
    y = np.linspace(*ylim,n)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack((X.flatten(), Y.flatten()))
    Ztar = np.ones((len(x), len(x)))
    if density is not None :
        Ztar = density.eval(pts).reshape((len(x), len(x)))
    if qcs is not None :
        qcs = ax.contourf(X, Y, Ztar, levels=qcs.levels, extend='both', cmap=cmap, alpha=alpha, zorder=1)
    else :
        qcs = ax.contourf(X, Y, Ztar, levels=nlevels, cmap=cmap, alpha=alpha, zorder=1)
    # grid
    if grid is not None :
        for line in grid :
            ax.plot(line[0], line[1], '#c8c8c8', lw=2, zorder=2)
    # samples
    if samples is not None :
        ax.scatter(samples[0], samples[1], s=7, color='#e41a1c', alpha=.9, zorder=3)

    plt.tight_layout()
    if filename is not None :
        plt.savefig('/home/uq/' + filename + '.pdf')
    return qcs, ax


def plot_tbs_results(t, s, p_uni, p_tar, lines, lines_t) :
    n = 4
    fig = plt.figure(figsize=(n*8,8))
    ax = lambda i : get_ax(fig=fig, nx=n, idx=i, square=True, xlabel='')

    qcs,_ = plot_density(density=t, ax=ax(1))
    plot_density(density=s, ax=ax(2), qcs=qcs)
    plot_density(density=s, ax=ax(3), qcs=qcs, grid=lines_t, samples=p_tar)
    plot_density(density=None, ax=ax(4), qcs=qcs, grid=lines, samples=p_uni)
    plt.tight_layout()
    plt.show()
