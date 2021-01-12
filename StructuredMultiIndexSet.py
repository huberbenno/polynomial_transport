import sys
sys.path.insert(0,'/home/uq/apps/muq2inst/lib')
import pymuqUtilities as mu

class StructuredMultiIndexSet :

    def __init__(self, multiset=None) :
        if multiset is not None :
            self.multilist = [multiset.IndexToMulti(i) for i in range(multiset.Size())]
            self.orders = multiset.GetMaxOrders()
            self.clusters = dict()
            self.groups = dict()
            self._init_clusters_recursive()

    def _init_clusters_recursive(self, current_indices=None, current_dim=None, prev=None) :
        if current_indices is None :
            current_indices = list(range(len(self.multilist)))
            prev = max([len(m.GetVector()) for m in self.multilist])
            current_dim = prev - 1

        tmp = dict()
        for idx in current_indices :
            mu_idx = self.multilist[idx].GetValue(current_dim)
            if mu_idx not in tmp.keys() :
                tmp[mu_idx] = [idx]
            else :
                tmp[mu_idx] += [idx]
        new_vals = [l for l in tmp.values()]
        n_old_vals = 0
        if current_dim not in self.clusters.keys() :
            self.clusters[current_dim] = new_vals
        else :
            n_old_vals = len(self.clusters[current_dim])
            self.clusters[current_dim] += new_vals

        if current_dim > 0 :
            for l in range(len(new_vals)) :
                self._init_clusters_recursive(new_vals[l], current_dim - 1, l+n_old_vals)

        if current_dim not in self.groups.keys() :
            self.groups[current_dim] = list()
        self.groups[current_dim].append(list(range(n_old_vals, n_old_vals + len(new_vals))))

    def test_22(self) :
        testset = mu.MultiIndexFactory.CreateTotalOrder(2,2)
        structuredset = StructuredMultiIndexSet(testset)

        for m in testset.GetAllMultiIndices() : print(m.GetVector())
        print(structuredset.clusters)
        print(structuredset.groups)

        assert(len(structuredset.clusters[0]) == 6)
        assert(len(structuredset.clusters[1]) == 3)
        for j in range(6) :
            assert(len(structuredset.clusters[0][j]) == 1)
        assert(len(structuredset.clusters[1][0]) == 3)
        assert(len(structuredset.clusters[1][1]) == 2)
        assert(len(structuredset.clusters[1][2]) == 1)

        assert(len(structuredset.groups[0]) == 3)
        assert(len(structuredset.groups[1]) == 1)

if __name__ == '__main__' :
    StructuredMultiIndexSet().test_22()
