import numpy as np
import itertools as it

import util
import Database as db


class MultiIndex :

    def __init__(self, d, *, sparse=None, dense=None) :
        assert (sparse is None) ^ (dense is None)
        self.d = d
        if dense :
            self.nzs = {}
            for k,v in enumerate(dense) :
                if v > 0 : self.nzs[k] = v
        else :
            self.nzs = sparse

    def __getitem__(self, i) :
        if i < 0 : i = self.d - i
        return self.nzs.get(i, 0)

    def asList(self) :
        dense = [0] * self.d
        for k,v in self.nzs.items() :
            dense[k] = v
        return dense

    def print(self) :
        print(self.asList())


# ---------- Indexsets --------------------

class MultiIndexSet :

    def __init__(self, *, name, dim, idxs, save=False) :
        self.name = name
        self.dim = dim
        self.idxs = idxs

        self.maxOrders = [0] * self.dim
        for idx in self.idxs :
            for k,v in idx.nzs.items() :
                self.maxOrders[k] = max(self.maxOrders[k], v)

        self.maxDegree = max(self.maxOrders)
        self.cardinality = len(self.idxs)
        self.weightsForLegendreL2Normalization = None

    def __getitem__(self, i) : return self.idxs[i]

    def asLists(self) : return [i.asList() for i in self.idxs]

    def print(self) :
        for idx in self.idxs :
            idx.print()

    def getWeightsForLegendreL2Normalization(self) :
        if self.weightsForLegendreL2Normalization is None :
            self.weightsForLegendreL2Normalization = np.array([np.prod([np.sqrt((2*l + 1)/2) for l in idx.asList()]) for idx in self.idxs])
        return self.weightsForLegendreL2Normalization

    def deleteDbo(self) :
        if hasattr(self, 'dbo') : self.dbo.delete_instance()


class TensorProductSet(MultiIndexSet) :

    def __init__(self, *, dim, order, save=False, verbose=False) :
        self.order = order

        idxs = [MultiIndex(dim, dense=idx) for idx in it.product(range(self.order+1), repeat=dim)]

        MultiIndexSet.__init__(self, name='tensorproduct', dim=dim, idxs=idxs, save=save)

        if save :
            self.dbo, _ = db.MultiIndexSetDBO.get_or_create(
                dim=dim, mode='tensorproduct', order=order, size=len(idxs))


class TotalDegreeSet(MultiIndexSet) :

    def __init__(self, *, dim, order, save=False, verbose=False) :
        self.order = order

        filtr = lambda x : sum(x) > order
        itr = it.product(range(order+1), repeat=dim)
        idxs = [MultiIndex(dim, dense=idx) for idx in it.filterfalse(filtr, itr)]

        MultiIndexSet.__init__(self, name='totaldegree', dim=dim, idxs=idxs, save=save)

        if save :
            self.dbo, _ = db.MultiIndexSetDBO.get_or_create(
                dim=dim, mode='totaldegree', order=order, size=len(idxs))


class AnisotropicSet(MultiIndexSet) :

    def __init__(self, *, weights, cardinality, save=False, verbose=False) :
        if verbose > 0 : print(f'\t AnisotropicSet with cardinality={cardinality} / ', end='')

        for i in range(len(weights)-1) :
            assert weights[i+1] > 0
            assert weights[i] <= weights[i+1]
        assert cardinality > 0

        # apply bisection to find set with suitable cardinality
        d = len(weights)
        k = weights / weights[0]
        f = lambda l : len(self._setup_idxs(k, l, cutoff=len(k)))
        l_interval = [1, 2]
        while f(l_interval[0]) > cardinality : l_interval[0] /= 1.2
        while f(l_interval[1]) < cardinality : l_interval[1] *= 1.2
        l = util.points.bisection(f, cardinality, interval=l_interval)
        idxs = [MultiIndex(d, sparse=idx) for idx in self._setup_idxs(k, l, cutoff=len(k))]

        MultiIndexSet.__init__(self, name='sparse', dim=d, idxs=idxs, save=save)

        if save :
            self.dbo, _ = db.MultiIndexSetAnisotropicDBO.get_or_create(
                dim=d, weight=db.to_string(k), thresh=l, size=len(idxs))
        if verbose > 0 : print('done.')

    def _setup_idxs(self, k, l, i=0, idx=None, *, cutoff=None) :
        if idx is None : idx = {}
        if cutoff is not None and i >= cutoff : return [idx]
        r = []
        if (cutoff is None or i+1 < cutoff) and k[i+1] < l :
            r += self._setup_idxs(k, l, i+1, idx, cutoff=cutoff)
        else :
            r += [idx]
        j = 1
        while(j * k[i] < l) :
            r += self._setup_idxs(k, l-j*k[i], i+1, {**idx, i : j}, cutoff=cutoff)
            j += 1
        return r


# ---------- Trees --------------------

class MultiIndexTreeNode :

    def __init__(self, idx: int, mlist: list) :
        self.idx = idx
        self.val = None
        self.children = []
        if len(mlist[0][1]) > 0 :
            mlist = sorted(mlist, key=lambda l : l[1][-1])
            for i, l in it.groupby(mlist, lambda l : l[1][-1]) :
                self.children.append(MultiIndexTreeNode(i, [(li[0],li[1][:-1]) for li in l]))

        else :
            self.val = mlist[0][0]

    def print(self) :
        print(self.idx, self.val, len(self.children))


class MultiIndexTree :

    def __init__(self, surrogate) :
        self.maxOrders = surrogate.multis.maxOrders
        self.root = MultiIndexTreeNode(None, list(zip(surrogate.coeffs, surrogate.multis.asLists())))
        self.nodes = [[]] * surrogate.multis.dim + [[self.root]]
        for i in range(surrogate.multis.dim, 0, -1) :
            self.nodes[i-1] = [*it.chain.from_iterable([n.children for n in self.nodes[i]])]

    def __getitem__(self, i) :
        return self.nodes[i]
