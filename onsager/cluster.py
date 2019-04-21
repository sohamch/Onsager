"""
Cluster expansion module: types necessary to implement cluster expansions of
quantities based on crystals.
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
import copy, collections, itertools, yaml
from onsager import crystal, supercell

# YAML tags
CLUSTERSITE_YAMLTAG = '!ClusterSite'
CLUSTER_YAMLTAG = '!Cluster'


class ClusterSite(collections.namedtuple('ClusterSite', 'ci R')):
    """
    A class corresponding to a site in a cluster.

    :param ci: (chem, index) of the site
    :param R: lattice vector of the site
    """

    def _asdict(self):
        """Return a proper dict"""
        return {'ci': self.ci, 'R': self.R}

    @classmethod
    def fromcryscart(cls, crys, cart_pos):
        """Return a ClusterSite corresponding to Cartesian position ``cart_pos`` in crystal ``crys``"""
        R, ci = crys.cart2pos(cart_pos)
        return cls(ci=ci, R=R)

    @classmethod
    def fromcrysunit(cls, crys, unit_pos):
        """Return a ClusterSite corresponding to unit cell position ``unit_pos`` in crystal ``crys``"""
        cart_pos = crys.unit2cart(np.zeros(crys.dim, dtype=int), unit_pos)
        return cls.fromcryscart(cart_pos)

    def __eq__(self, other):
        """Test for equality--we don't bother checking dx"""
        return isinstance(other, self.__class__) and \
               (self.ci == other.ci and np.all(self.R == other.R))

    def __ne__(self, other):
        """Inequality == not __eq__"""
        return not self.__eq__(other)

    def __hash__(self):
        """Hash, so that we can make sets of states"""
        # return self.i ^ (self.j << 1) ^ (self.R[0] << 2) ^ (self.R[1] << 3) ^ (self.R[2] << 4)
        return hash(self.ci + tuple(self.R))

    def __neg__(self):
        """Negation of site"""
        return self.__class__(ci=self.ci, R=-self.R)

    def __add__(self, other):
        """Add a vector to a site; other *must* be a vector
        """
        if len(other) != len(self.R):
            raise ArithmeticError('Dimensionality problem? Adding {} to {}'.format(other, self))
        return self.__class__(ci=self.ci, R=self.R + np.array(other))

    def __sub__(self, other):
        return self.__add__(-np.array(other))

    def g(self, crys, g):
        """
        Apply group operation.

        :param crys: crystal
        :param g: group operation (from crys)
        :return g*site: corresponding to group operation applied to self
        """
        gR, gci = crys.g_pos(g, self.R, self.ci)
        return self.__class__(ci=gci, R=gR)

    def __str__(self):
        """Human readable version"""
        if len(self.R) == 3:
            return "{}.[{},{},{}]".format(self.ci, self.R[0], self.R[1], self.R[2])
        else:
            return "{}.[{},{}]".format(self.ci, self.R[0], self.R[1])

    @staticmethod
    def ClusterSite_representer(dumper, data):
        """Output a ClusterSite"""
        # asdict() returns an OrderedDictionary, so pass through dict()
        # had to rewrite _asdict() for some reason...?
        return dumper.represent_mapping(CLUSTERSITE_YAMLTAG, data._asdict())

    @staticmethod
    def ClusterSite_constructor(loader, node):
        """Construct a ClusterSite from YAML"""
        # ** turns the dictionary into parameters for ClusterSite constructor
        return ClusterSite(**loader.construct_mapping(node, deep=True))


yaml.add_representer(ClusterSite, ClusterSite.ClusterSite_representer)
yaml.add_constructor(CLUSTERSITE_YAMLTAG, ClusterSite.ClusterSite_constructor)


class Cluster(object):
    """
    Class to define (arbitrary) cluster interactions. A cluster is defined as
    a set of ClusterSites. We don't implement this using sets, however, because
    we make a choice of a "reference" site in each cluster, which has a lattice
    vector of 0, to account for translational invariance.

    The flag transition dictates whether the cluster is a transition state cluster
    or not. The difference with a transition state cluster is that the first two
    states in the cluster are the initial and final states of the transition, while
    all of the remaining parts are the cluster.
    """

    def __init__(self, clustersitelist, transition=False, NOSORT=False):
        """
        Cluster interaction, from an iterable of ClusterSites
        :param clustersitelist: iterable of ClusterSites
        :param transition: True if a transition state cluster; cl[0] = initial, cl[1] = final
        """
        # this sorting is a *little* hacked together, but as long as we don't have
        # more than 2^32 sites in our crystal, we're good to go:
        def sortkey(cs): return cs.ci[0]*(2**32)+cs.ci[1]
        # first, dump contents of iterable into a list to manipulate:
        lis = [cs for cs in clustersitelist]
        if not NOSORT:
            if not transition:
                lis.sort(key=sortkey)
            else:
                lis = lis[0:2] + sorted(lis[2:], key=sortkey)
        R0 = lis[0].R
        self.sites = tuple([cs-R0 for cs in lis])
        self.Norder = len(self.sites)
        self.Nsites = len(self.sites)
        if transition:
            self.Norder -= 2
        # a little mapping of the positions into sets to make equality checking faster,
        # and explicit evaluation of hash function one time using XOR of individual values
        # so that it respects permutations
        self.__center__ = sum(cs.R for cs in self.sites)
        self.__equalitymap__ = {}
        hashcache = 0
        for cs in self.sites:
            shiftpos = self.__shift_pos__(cs)  # our tuple representation of site
            hashcache ^= hash(cs.ci + shiftpos)
            if cs.ci not in self.__equalitymap__:
                self.__equalitymap__[cs.ci] = set([shiftpos])
            else:
                self.__equalitymap__[cs.ci].add(shiftpos)
        self.__hashcache__ = hashcache
        self.__transition__ = transition

    def __shift_pos__(self, cs):
        return tuple(cs.R*self.Nsites - self.__center__)

    def __eq__(self, other):
        """
        Test for equality of two clusters. This is a bit trickier than one would expect, since
        clusters are essentially sets where all of the lattice vectors are shifted by the
        center of mass of the sites.
        """
        if not isinstance(other, self.__class__): return False
        if self.__transition__ != other.__transition__: return False
        if self.Norder != other.Norder: return False
        if self.__equalitymap__.keys() != other.__equalitymap__.keys(): return False
        for k, v in self.__equalitymap__.items():
            if other.__equalitymap__[k] != v: return False
        if self.__transition__:
            # TSself, TSother = self.transitionstate(), other.transitionstate()
            # if TSself != TSother:
            #     R0 = TSother[1].R
            #     if TSself != (TSother[1] - R0, TSother[0] - R0):
            if not self.istransition(*other.transitionstate()):
                return False
        return True

    def __hash__(self):
        """Return our hash value, precomputed"""
        return self.__hashcache__

    def __contains__(self, elem):
        if elem.ci not in self.__equalitymap__: return False
        return self.__shift_pos__(elem) in self.__equalitymap__[elem.ci]

    def __getitem__(self, item):
        if item >= self.Norder: raise IndexError
        if self.__transition__:
            return self.sites[2:][item]
        else:
            return self.sites[item]

    def _asdict(self):
        """Return a proper dict"""
        if self.__transition__:
            return {'clustersitelist': self.sites, 'transition': self.__transition__}
        else:
            return {'clustersitelist': self.sites}

    def transitionstate(self):
        """Return the two sites of the transition state"""
        if not self.__transition__:
            raise ValueError('Not a TS cluster')
        return self.sites[0], self.sites[1]

    def istransition(self, site0, site1):
        """Check whether two sites correspond to the transition state"""
        if not self.__transition__:
            raise ValueError('Not a TS cluster')
        R0 = site0.R
        if self.sites[0] == site0 - R0 and self.sites[1] == site1 - R0:
            return True
        R1 = site1.R
        if self.sites[0] == site1 - R1 and self.sites[1] == site0 - R1:
            return True
        return False

    def __add__(self, other):
        """Add a clustersite to a cluster expansion"""
        if other not in self:
            return self.__class__(self.sites + (other,), transition=self.__transition__)
        else:
            return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract a site from a cluster. Had a very specific meaning: if the
        site is in the cluster, it returns a list of all the *other* sites, shifted
        so that the ``other`` site is at the origin.

        Needs to be clarified for the case of a transition state.

        :param other: needs to be a cluster site *in* the cluster.
        """
        if self.__transition__:
            raise NotImplemented('Subtraction not currently implemented for TS clusters')
        if other not in self:
            raise ArithmeticError('{} not in {}'.format(other, self))
        return [cs-other.R for cs in self.sites if cs != other]

    def __len__(self):
        return self.Norder

    def g(self, crys, g):
        """
        Apply group operation.

        :param crys: crystal
        :param g: group operation (from crys)
        :return g*cluster: corresponding to group operation applied to self
        """
        return self.__class__([cs.g(crys, g) for cs in self.sites], transition=self.__transition__)

    def pairdistances(self, crys):
        """
        Return a dictionary of all the pair distances between the chemistries in the cluster.
        For simplicity, we include both (c1,c2) and (c2,c1).

        :param crys: crystal
        :return dist_dict: dist_dict[c1][c2] = sorted list of distances between chemistry c1 and c2
        """
        chem_cart = [(cs.ci[0], crys.pos2cart(cs.R, cs.ci)) for cs in self.sites]
        dist_dict = {}
        for n0, (c0, x0) in enumerate(chem_cart):
            for (c1, x1) in chem_cart[:n0]:
                d = np.sqrt(np.dot(x1-x0, x1-x0))
                tup = (c0, c1) if c0 <= c1 else (c1, c0)
                if tup in dist_dict:
                    dist_dict[tup].append(d)
                else:
                    dist_dict[tup] = [d]
        # sort:
        for dlist in dist_dict.values():
            dlist.sort()
        # put (c1,c0) in place: (needs a list comprehension to avoid runtime error about adding keys while iterating]
        for (c0, c1) in [tup for tup in dist_dict.keys()]:
            if (c1, c0) not in dist_dict:
                dist_dict[c1, c0] = dist_dict[c0, c1]
        return dist_dict

    def __str__(self):
        """Human readable version"""
        s = "{} order: ".format(self.Norder)
        if self.__transition__:
            s += str(self.sites[0]) + " -> " + str(self.sites[1]) + " : "
            s += " ".join([str(cs) for cs in self.sites[2:]])
        else:
            s += " ".join([str(cs) for cs in self.sites])
        return s

    @staticmethod
    def Cluster_representer(dumper, data):
        """Output a ClusterSite"""
        # asdict() returns an OrderedDictionary, so pass through dict()
        # had to rewrite _asdict() for some reason...?
        return dumper.represent_mapping(CLUSTER_YAMLTAG, data._asdict())

    @staticmethod
    def Cluster_constructor(loader, node):
        """Construct a ClusterSite from YAML"""
        # ** turns the dictionary into parameters for ClusterSite constructor
        return Cluster(**loader.construct_mapping(node, deep=True))

yaml.add_representer(Cluster, Cluster.Cluster_representer)
yaml.add_constructor(CLUSTER_YAMLTAG, Cluster.Cluster_constructor)


def makeclusters(crys, cutoff, maxorder, exclude=()):
    """
    Function to make clusters up to a maximum order involving all sites within a cutoff
    distance. We can exclude certain chemistries; default is to use all.

    :param crys: crystal to construct our clusters for
    :param cutoff: distance between sites; all sites in a cluster must
      have this mutual distance
    :param maxorder: maximum order of our clusters
    :param exclude: list of chemistries to exclude
    :return clusterexp: list of sets of clusters
    """
    sitelist = [ci for ci in crys.atomindices if ci[0] not in exclude]
    # We construct our clusters in increasing order for maximum efficiency
    # 1st order (sites) is slightly different than the rest.
    clusterexp = []
    clusters = set()
    for ci in sitelist:
        # single sites:
        cl = Cluster([ClusterSite(ci, np.zeros(crys.dim, dtype=int))])
        if cl not in clusters:
            clset = set([cl.g(crys, g) for g in crys.G])
            clusterexp.append(clset)
            clusters.update(clset)
    if maxorder < 2:
        return clusterexp
    # now, we can proceed to higher and higher orders...
    # first, make lists of all our pairs within a given nn distance
    # we could modify this to use different cutoff between different chemistries...
    r2 = cutoff * cutoff
    nmax = [int(np.round(np.sqrt(r2/crys.metric[i, i]))) + 1
            for i in range(crys.dim)]
    nranges = [range(-n, n+1) for n in nmax]
    supervect = [np.array(ntup) for ntup in itertools.product(*nranges)]
    nndict = {}
    for ci0 in sitelist:
        u0 = crys.basis[ci0[0]][ci0[1]]
        nnset = set()
        for ci1 in sitelist:
            u1 = crys.basis[ci1[0]][ci1[1]]
            du = u1 - u0
            for R in supervect:
                dx = crys.unit2cart(R, du)
                if 0 < np.dot(dx, dx) < r2:
                    nnset.add(ClusterSite(ci1, R))
        nndict[ci0] = nnset
    for K in range(maxorder-1):
        # we build based on our lower order clusters:
        prevclusters, clusters = clusters, set()
        for clprev in prevclusters:
            for neigh in nndict[clprev[0].ci]:
                # if this neighbor is already in our cluster, move on
                if neigh in clprev: continue
                # now check that all of the sites in the cluster are also neighbors:
                neighlist = nndict[neigh.ci]
                R0 = neigh.R
                if all(ClusterSite(cl.ci, cl.R-R0) in neighlist for cl in clprev):
                    # new cluster!
                    clnew = clprev + neigh
                    if clnew not in clusters:
                        clset = set([clnew.g(crys, g) for g in crys.G])
                        clusterexp.append(clset)
                        clusters.update(clset)
    return clusterexp

def makeTSclusters(crys, chem, jumpnetwork, clusterexp):
    """
    Function to make TS clusters based on an existing cluster expansion corresponding
    to a given jump network.

    :param crys: crystal to construct our clusters for
    :param chem: index of mobile species
    :param jumpnetwork: list of lists of ((i, j), dx) transitions
    :param clusterexp: list of sets of clusters to base TS cluster expansion
    :return TSclusterexp: list of sets of TS clusters
    """
    # convert the entire chem / jumpnetwork into pairs of sites:
    jumppairs = []
    for jn in jumpnetwork:
        for ((i, j), dx) in jn:
            R = np.round(np.dot(crys.invlatt, dx) - crys.basis[chem][j] + crys.basis[chem][i]).astype(int)
            jumppairs.append((ClusterSite((chem, i), np.zeros(crys.dim, dtype=int)), ClusterSite((chem, j), R)))
    TSclusterexp = []
    TSclusters = set()
    # we run through the clusters in the order they appear in the cluster expansion,
    # so that if clusters are in increasing order, then they will be when returned
    for clustlist in clusterexp:
        if sum(1 for site in next(iter(clustlist)) if site.ci[0] == chem) < 2: continue
        # we can only use clusters that have at least two mobile sites
        for clust in clustlist:
            for cs_i, cs_j in jumppairs:
                for site in clust:
                    if site.ci == cs_i.ci:
                        cl_list = clust - site
                        if cs_j in cl_list:
                            cl_list.remove(cs_j) # remove in place
                            TSclust = Cluster([cs_i, cs_j] + cl_list, transition=True)
                            if TSclust not in TSclusters:
                                # new transition state cluster
                                TSclset = set([TSclust.g(crys, g) for g in crys.G])
                                TSclusterexp.append(TSclset)
                                TSclusters.update(TSclset)
    return TSclusterexp


class MonteCarloSampler(object):
    """
    An object to maintain state in a supercell, evaluate energies efficiently including
    "trial" moves. Built from cluster expansions and using a cluster supercell.
    """
    def __init__(self, supercell, spectator_occ, clusterexp, enevalues,
                 chem=None, jumpnetwork=(), KRAvalues=0, TSclusters=(), TSvalues=()):
        """
        Setup a MonteCarloSampler using a supercell, with a given spectator occupancy,
        cluster expansion, and energy values for the clusters. Now includes the ability to
        evaluate a jumpnetwork, which is optional. Because we need to be consistent with
        our cluster expansion, can only be done at initialization.

        :param supercell: should be a ClusterSupercell
        :param spectator_occ: vector of occupancies for spectator species (0 or 1),
          consistent with our supercell
        :param clusterexp: list of sets of cluster interactions
        :param enevalues: energy values corresponding to each cluster

        :param chem: (optional) index of species that transitions
        :param jumpnetwork: (optional) list of lists of jumps; each is ((i, j), dx) where ``i`` and ``j`` are
          unit cell indices for species ``chem``
        :param KRAvalues: (optional) list of "KRA" values for barriers (relative to average energy of endpoints);
          if ``TSclusters`` are used, choosing 0 is more straightforward.
        :param TSclusters: (optional) list of transition state cluster expansion terms; this is
          always added on to KRAvalues (thus using 0 is recommended if TSclusters are also used)
        :param TSvalues: (optional) values for TS cluster expansion entries
        """
        self.supercell = supercell
        siteinteract, interactvalue = supercell.clusterevaluator(spectator_occ, clusterexp, enevalues)
        self.Nenergy = len(interactvalue)
        # to be initialized via jumpnetwork_init()
        self.jumps = None  # indicates no jump network...
        if chem is not None:
            # quick check that chem is a mobile species:
            if (chem, 0) not in self.supercell.indexmobile:
                raise ValueError('Chemical species {} is a spectator in supercell?'.format(chem))
            siteinteract, interactvalue, self.jumps, self.interactrange = \
                self.supercell.jumpnetworkevaluator(spectator_occ, clusterexp, enevalues, chem, jumpnetwork,
                                                    KRAvalues, TSclusters, TSvalues, siteinteract, interactvalue)
        # convert from lists to arrays:
        self.Ninteract = np.array([len(inter) for inter in siteinteract])
        # see https://stackoverflow.com/questions/38619143/convert-python-sequence-to-numpy-array-filling-missing-values
        self.siteinteract = np.array(list(itertools.zip_longest(*siteinteract, fillvalue=-1))).T
        self.interactvalue = np.array(interactvalue)
        # to be initialized with start()
        self.occ, self.clustercount, self.occupied_set, self.unoccupied_set = None, None, None, None

    def start(self, occ):
        """
        Initialize with an occupancy, and prepare for future calculations.

        :param occ: occupancy of sites in supercell; assumed to be 0 or 1
        """
        # NOTE: we don't do this with a copy() operation...
        self.occ = occ
        self.clustercount = np.zeros_like(self.interactvalue, dtype=int)
        occ_list, unocc_list = [], []
        for i, occ_i, interact, Ninteract in zip(itertools.count(), self.occ, self.siteinteract, self.Ninteract):
            if occ_i == 0:
                unocc_list.append(i)
                for m in interact[:Ninteract]:
                    self.clustercount[m] += 1
                # for n in range(Ninteract):
                #     self.clustercount[interact[n]] += 1
            else:
                occ_list.append(i)
        self.occupied_set = set(occ_list)
        self.unoccupied_set = set(unocc_list)

    def E(self):
        """
        Compute the energy.

        :return E: total of all interactions
        """
        E = 0
        for ccount, Evalue in zip(self.clustercount[:self.Nenergy], self.interactvalue[:self.Nenergy]):
            if ccount == 0:
                E += Evalue
        return E

    def transitions(self):
        """
        Compute all transitions.

        :return ijlist: list of (initial, final) tuples for each transition
        :return Qlist: vector of energy barriers for each transition
        :return dxlist: vector of displacements for each transition
        """
        if self.jumps is None:
            raise ValueError('No jump network in sampler.')
        ijlist, Qlist, dxlist = [], [], []

        for n, ((i, j), dx) in enumerate(self.jumps):
            if self.occ[i] == 0 or self.occ[j] == 1:
                continue
            ijlist.append((i, j))
            dxlist.append(dx)
            ran = slice(self.interactrange[n - 1], self.interactrange[n])
            Qlist.append(sum(E for E, c in zip(self.interactvalue[ran], self.clustercount[ran]) if c == 0))
        return ijlist, np.array(Qlist), np.array(dxlist)

    def deltaE_trial(self, occsites=(), unoccsites=()):
        """
        Compute the energy change if the sites in occsites are occupied, and the sites in
        unoccsites are unoccupied.

        A few notes: the algorithm does not check whether the same site appears in
        either iterable multiple times; it trusts that the user has provided it with
        a meaningful trial change.

        :param occsites: iterable of sites to attempt occupying
        :param unoccsites: iterable of sites to attempt unoccupying
        :return deltaE: change in energy
        """
        # we're going to keep track just of the interactions that we change;
        # this change will be kept in a dictionary, and will be the *negative* of the
        # clustercount change that would occur with the trial move
        dclustercount = {}
        for i in occsites:
            if self.occ[i] == 0:
                for inter in self.siteinteract[i][:self.Ninteract[i]]:
                    if inter in dclustercount:
                        dclustercount[inter] += 1
                    else:
                        dclustercount[inter] = 1
        for i in unoccsites:
            if self.occ[i] == 1:
                for inter in self.siteinteract[i][:self.Ninteract[i]]:
                    if inter in dclustercount:
                        dclustercount[inter] -= 1
                    else:
                        dclustercount[inter] = -1
        dE = 0
        for interact, dcount in dclustercount.items():
            # no change?
            if dcount == 0: continue
            # not an *energy* interaction?
            if interact >= self.Nenergy: continue
            # are we turning off an interaction?
            if self.clustercount[interact] == 0:
                dE -= self.interactvalue[interact]
            # are we turning on an interaction?
            elif self.clustercount[interact] == dcount:
                dE += self.interactvalue[interact]
        return dE

    def update(self, occsites=(), unoccsites=()):
        """
        Update the state to occupy the sites in occsites and un-occupy the sites in unoccsites.

        :param occsites: iterable of sites to occupy
        :param unoccsites: iterable of sites to unoccupy
        """
        for i in occsites:
            if self.occ[i] == 0:
                self.occ[i] = 1
                self.unoccupied_set.remove(i)
                self.occupied_set.add(i)
                for inter in self.siteinteract[i][:self.Ninteract[i]]:
                    self.clustercount[inter] -= 1
        for i in unoccsites:
            if self.occ[i] == 1:
                self.occ[i] = 0
                self.unoccupied_set.add(i)
                self.occupied_set.remove(i)
                for inter in self.siteinteract[i][:self.Ninteract[i]]:
                    self.clustercount[inter] += 1

from numba import jitclass          # import the decorator
from numba import int64, float64    # import the types

# our signature for our object
MonteCarloSamplerSpec = [
    ('Nenergy', int64),
    ('Njumps', int64),
    ('jump_ij', int64[:, :]),
    ('jump_dx', float64[:, :]),
    ('jump_Q', float64[:]),
    ('interactrange', int64[:]),
    ('Ninteract', int64[:]),
    ('siteinteract', int64[:, :]),
    ('interactvalue', float64[:]),
    ('Nsites', int64),
    ('occ', int64[:]),
    ('clustercount', int64[:]),
    ('dcluster', int64[:]),
    ('Nocc', int64),
    ('Nunocc', int64),
    ('occupied_set', int64[:]),
    ('unoccupied_set', int64[:]),
    ('index', int64[:])
]

@jitclass(MonteCarloSamplerSpec)
class MonteCarloSampler_jit(object):
    """
    Numba jit wrapper on a MonteCarloSampler.
    """
    def __init__(self, MCsampler):
        """
        Setup a jit-version of a MonteCarloSampler from an existing one.

        :param MCsampler: Monte Carlo sampler to extract all of the current information from.
        """
        self.Nenergy = MCsampler.Nenergy
        # to be initialized via jumpnetwork_init()
        self.Njumps = 0
        self.jump_ij = np.zeros((self.Njumps,2), dtype=int)  # indicates no jump network...
        self.jump_dx = np.zeros((self.Njumps,3), dtype=float)
        self.jump_Q = np.zeros(self.Njumps, dtype=float)
        self.interactrange = np.zeros(0, dtype=int)
        if MCsampler.jumps is not None:
            self.Njumps = len(MCsampler.jumps)
            self.jump_ij = np.array([[i,j] for (i, j), _ in MCsampler.jumps])
            self.jump_dx = np.array([dx for _, dx in MCsampler.jumps])
            self.jump_Q = np.zeros(self.Njumps)
            self.interactrange = np.array(MCsampler.interactrange)
        # convert from lists to arrays:
        self.Ninteract = MCsampler.Ninteract
        # see https://stackoverflow.com/questions/38619143/convert-python-sequence-to-numpy-array-filling-missing-values
        self.siteinteract = MCsampler.siteinteract
        self.interactvalue = MCsampler.interactvalue
        # to be initialized with start()
        self.Nsites = MCsampler.supercell.size*MCsampler.supercell.Nmobile
        self.occ = MCsampler.occ.copy()
        self.clustercount = MCsampler.clustercount.copy()
        self.dcluster = np.zeros(self.Nenergy, dtype=int)
        self.Nocc = 0
        self.Nunocc = 0
        self.occupied_set = np.zeros(len(self.occ), dtype=int)
        self.unoccupied_set = np.zeros(len(self.occ), dtype=int)
        self.index = np.zeros(len(self.occ), dtype=int)
        for i in range(len(self.occ)):
            if self.occ[i] == 1:
                self.occupied_set[self.Nocc] = i
                self.index[i] = self.Nocc
                self.Nocc += 1
            else:
                self.unoccupied_set[self.Nunocc] = i
                self.index[i] = self.Nunocc
                self.Nunocc += 1

    def start(self, occ):
        """
        Initialize with an occupancy, and prepare for future calculations.

        :param occ: occupancy of sites in supercell; assumed to be 0 or 1
        """
        # start from scratch:
        self.clustercount[:] = 0
        self.Nocc = 0
        self.Nunocc = 0
        # Note: now we keep our own copy of occ internally
        for i in range(self.Nsites):
            self.occ[i] = occ[i]
            if occ[i] == 1:
                self.occupied_set[self.Nocc] = i
                self.index[i] = self.Nocc
                self.Nocc += 1
            else:
                self.unoccupied_set[self.Nunocc] = i
                self.index[i] = self.Nunocc
                self.Nunocc += 1
                for n in range(self.Ninteract[i]):
                    self.clustercount[self.siteinteract[i, n]] += 1

    def E(self):
        """
        Compute the energy.

        :return E: total of all interactions
        """
        E = 0
        for n in range(self.Nenergy):
            if self.clustercount[n] == 0:
                E += self.interactvalue[n]
        return E

    def transitions(self):
        """
        Compute all transitions.

        :return ijlist: vector of (initial, final) tuples for each transition
        :return Qlist: vector of energy barriers for each transition (Inf == forbidden)
        :return dxlist: vector of displacements for each transition
        """
        for n in range(self.Njumps):
            if self.occ[self.jump_ij[n][0]] == 0 or self.occ[self.jump_ij[n][1]] == 1:
                self.jump_Q[n] = np.Inf
            else:
                self.jump_Q[n] = 0.
                for m in range(self.interactrange[n - 1], self.interactrange[n]):
                    if self.clustercount[m] == 0:
                        self.jump_Q[n] += self.interactvalue[m]
        return self.jump_ij, self.jump_Q, self.jump_dx

    def deltaE_trial(self, occsite, unoccsite):
        """
        Compute the energy change for swapping two sites.

        Note: this is less general than our non-jit version, to see if we can be
        more efficient. Note also: it does NOT check if the two sites are currently
        occupied or not. The behavior is unspecified for incorrect inputs.

        :param occsite: single site to occupy
        :param unoccsite: single site to unoccupy
        :return deltaE: change in energy
        """
        # we're going to keep track just of the interactions that we change;
        # this change will be kept in a dictionary, and will be the *negative* of the
        # clustercount change that would occur with the trial move
        self.dcluster[:] = 0
        for m in range(self.Ninteract[occsite]):
            n = self.siteinteract[occsite, m]
            if n >= self.Nenergy: break
            self.dcluster[n] += 1
        for m in range(self.Ninteract[unoccsite]):
            n = self.siteinteract[unoccsite, m]
            if n >= self.Nenergy: break
            self.dcluster[n] -= 1
        dE = 0
        for n in range(self.Nenergy):
            if self.dcluster[n] == 0: continue
            # are we turning off an interaction?
            if self.clustercount[n] == 0:
                dE -= self.interactvalue[n]
            # are we turning on an interaction?
            elif self.clustercount[n] == self.dcluster[n]:
                dE += self.interactvalue[n]
        return dE

    def update(self, occsite, unoccsite):
        """
        Update the state to occupy the site in occsite and un-occupy the site in unoccsite.

        :param occsite: site to occupy
        :param unoccsite: site to unoccupy
        """
        # change the occupancies:
        self.occ[occsite] = 1
        self.occ[unoccsite] = 0
        # change the cluster counts:
        for m in range(self.Ninteract[occsite]):
            self.clustercount[self.siteinteract[occsite, m]] -= 1
        for m in range(self.Ninteract[unoccsite]):
            self.clustercount[self.siteinteract[unoccsite, m]] += 1
        # change the "sets":
        i = self.index[occsite]  # index of occsite in unoccupied_set
        j = self.index[unoccsite]  # index of unoccsite in occupied_set
        self.occupied_set[j] = occsite
        self.unoccupied_set[i] = unoccsite
        self.index[occsite] = j  # index of occsite in occupied_set
        self.index[unoccsite] = i  # index of unoccsite in unoccupied_set
