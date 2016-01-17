"""
Onsager calculator module: Interstitialcy mechanism and Vacancy-mediated mechanism

Class to create an Onsager "calculator", which brings two functionalities:
1. determines *what* input is needed to compute the Onsager (mobility, or L) tensors
2. constructs the function that calculates those tensors, given the input values.

This class is designed to be combined with code that can, e.g., automatically
run some sort of atomistic-scale (DFT, classical potential) calculation of site
energies, and energy barriers, and then in concert with scripts to convert such data
into rates and probabilities, this will allow for efficient evaluation of transport
coefficients.

This implementation will be for vacancy-mediated solute diffusion assumes the dilute limit.
The mathematics is based on a Green function solution for the vacancy diffusion. The
computation of the GF is included in the GFcalc module.

Now with HDF5 write / read capability for VacancyMediated module
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
from scipy.linalg import pinv2, solve
from . import GFcalc
from . import crystal
from . import crystalStars as stars
from functools import reduce
import copy
import collections
import itertools


class Interstitial(object):
    """
    A class to compute interstitial diffusivity; uses structure of crystal to do most
    of the heavy lifting in terms of symmetry, etc.
    """
    def __init__(self, crys, chem, sitelist, jumpnetwork):
        """
        Initialization; takes an underlying crystal, a choice of atomic chemistry,
        a corresponding Wyckoff site list and jump network.

        :param crys : Crystal object
        :param chem : integer, index into the basis of crys, corresponding to the chemical element that hops
        :param sitelist : list of lists of indices, site indices where the atom may hop;
          grouped by symmetry equivalency
        :param jumpnetwork : list of lists of tuples: ( (i, j), dx )
            symmetry unique transitions; each list is all of the possible transitions
            from site i to site j with jump vector dx; includes i->j and j->i
        """
        self.crys = crys
        self.chem = chem
        self.sitelist = sitelist
        self.N = sum(1 for w in sitelist for i in w)
        self.invmap = [0 for w in sitelist for i in w]
        for ind,w in enumerate(sitelist):
            for i in w:
                self.invmap[i] = ind
        self.jumpnetwork = jumpnetwork
        self.VectorBasis, self.VV = self.generateVectorBasis()
        self.NV = len(self.VectorBasis)
        # quick check to see if our projected omega matrix will be invertible
        # only really needed if we have a non-empty vector basis
        self.omega_invertible = True
        if self.NV > 0:
            # invertible if inversion is present
            self.omega_invertible = any( np.allclose(g.cartrot, -np.eye(3)) for g in crys.G )
        if self.omega_invertible:
            # invertible, so just use solve for speed (omega is technically *negative* definite)
            self.bias_solver = lambda omega,b: -solve(-omega, b, sym_pos=True)
        else:
            # pseudoinverse required:
            self.bias_solver = lambda omega,b: np.dot(pinv2(omega), b)
        # these pieces are needed in order to compute the elastodiffusion tensor
        self.sitegroupops = self.generateSiteGroupOps() # list of group ops to take first rep. into whole list
        self.jumpgroupops = self.generateJumpGroupOps() # list of group ops to take first rep. into whole list
        self.siteSymmTensorBasis = self.generateSiteSymmTensorBasis() # projections for *first rep. only*
        self.jumpSymmTensorBasis = self.generateJumpSymmTensorBasis() # projections for *first rep. only*

    @staticmethod
    def sitelistYAML(sitelist):
        """Dumps a "sample" YAML formatted version of the sitelist with data to be entered"""
        return crystal.yaml.dump({'Dipole': [np.zeros((3,3)) for w in sitelist],
                                  'Energy': [0 for w in sitelist],
                                  'Prefactor': [1 for w in sitelist],
                                  'sitelist': sitelist})

    @staticmethod
    def jumpnetworkYAML(jumpnetwork):
        """Dumps a "sample" YAML formatted version of the jumpnetwork with data to be entered"""
        return crystal.yaml.dump({'DipoleT': [np.zeros((3,3)) for t in jumpnetwork],
                                  'EnergyT': [0 for t in jumpnetwork],
                                  'PrefactorT': [1 for t in jumpnetwork],
                                  'jumpnetwork': jumpnetwork})

    def generateVectorBasis(self):
        """
        Generate our full vector basis, using the information from our crystal
        :return: list of our unique vector basis lattice functions, normalized
        """
        def vectlist(vb):
            """Returns a list of orthonormal vectors corresponding to our vector basis
            :param vb: (dim, v)
            :return: list of vectors
            """
            if vb[0] == 0: return []
            if vb[0] == 1: return [vb[1]]
            if vb[0] == 2:
                # now, construct the other two directions:
                norm = vb[1]
                if abs(norm[2]) < 0.75:
                    v1 = np.array([norm[1], -norm[0], 0])
                else:
                    v1 = np.array([-norm[2], 0, norm[0]])
                v1 /= np.sqrt(np.dot(v1, v1))
                v2 = np.cross(norm, v1)
                return [v1, v2]
            if vb[0] == 3: return [np.array([1.,0.,0.]),
                                   np.array([0.,1.,0.]),
                                   np.array([0.,0.,1.])]

        lis = []
        for s in self.sitelist:
            for v in vectlist(self.crys.VectorBasis((self.chem, s[0]))):
                v /= np.sqrt(len(s)) # additional normalization
                # we have some constructing to do... first, make the vector we want to use
                vb = np.zeros((self.N, 3))
                for g in self.crys.G:
                    # what site do we land on, and what's the vector? (this is slight overkill)
                    vb[g.indexmap[self.chem][s[0]]] = self.crys.g_direc(g, v)
                lis.append(vb)
        # need the *full matrix of this tensor*
        VV = np.zeros((3,3,len(lis),len(lis)))
        for i,vb_i in enumerate(lis):
            for j,vb_j in enumerate(lis):
                VV[:,:,i,j] = np.dot(vb_i.T, vb_j)
        return lis, VV

    def generateSiteGroupOps(self):
        """
        Generates a list of group operations that transform the first site in each site list
        into all of the other members
        :return: list of list of group ops that mirrors the structure of site list
        """
        groupops = []
        for sites in self.sitelist:
            i0 = sites[0]
            oplist = []
            for i in sites:
                for g in self.crys.G:
                    if g.indexmap[self.chem][i0] == i:
                        oplist.append(g)
                        break
            groupops.append(oplist)
        return groupops

    def generateJumpGroupOps(self):
        """
        Generates a list of group operations that transform the first jump in the jump
        network into all of the other members
        :return: list of list of group ops that mirrors the structure of jumpnetwork
        """
        groupops = []
        for jumps in self.jumpnetwork:
            (i0,j0), dx0 = jumps[0]
            oplist = []
            for (i,j), dx in jumps:
                for g in self.crys.G:
                    # more complex: have to check the tuple (i,j) *and* the rotation of dx
                    # AND against the possibility that we are looking at the reverse jump too
                    if (g.indexmap[self.chem][i0] == i
                        and g.indexmap[self.chem][j0] == j
                        and np.allclose(dx, np.dot(g.cartrot, dx0)) ) or \
                            (g.indexmap[self.chem][i0] == j
                             and g.indexmap[self.chem][j0] == i
                             and np.allclose(dx, -np.dot(g.cartrot, dx0)) ):
                        oplist.append(g)
                        break
            groupops.append(oplist)
        return groupops

    def generateSiteSymmTensorBasis(self):
        """
        Generates a list of symmetric tensor bases for the first representative site
        in our site list.
        :return: list of symmetric bases
        """
        return [self.crys.SymmTensorBasis((self.chem, sites[0])) for sites in self.sitelist]

    def generateJumpSymmTensorBasis(self):
        """
        Generates a list of symmetric tensor bases for the first representative transition
        in our jump network
        :return: list of symmetric bases
        """
        # there is probably another way to do a list comprehension here, but that
        # will likely be nigh unreadable.
        lis = []
        for jumps in self.jumpnetwork:
            (i,j), dx = jumps[0]
            # more complex: have to check the tuple (i,j) *and* the rotation of dx
            # AND against the possibility that we are looking at the reverse jump too
            lis.append(reduce(crystal.CombineTensorBasis,
                              [crystal.SymmTensorBasis(*g.eigen())
                               for g in self.crys.G
                               if (g.indexmap[self.chem][i] == i and
                                   g.indexmap[self.chem][j] == j and
                                   np.allclose(dx, np.dot(g.cartrot, dx)) ) or
                               (g.indexmap[self.chem][i] == j and
                                g.indexmap[self.chem][j] == i and
                                np.allclose(dx, -np.dot(g.cartrot, dx)) ) ] ) )
        return lis

    def siteprob(self, pre, betaene):
        """Returns our site probabilities, normalized, as a vector"""
        # be careful to make sure that we don't under-/over-flow on beta*ene
        minbetaene = min(betaene)
        rho = np.array([ pre[w]*np.exp(minbetaene-betaene[w]) for w in self.invmap])
        return rho/sum(rho)

    def ratelist(self, pre, betaene, preT, betaeneT):
        """Returns a list of lists of rates, matched to jumpnetwork"""
        # the ij tuple in each transition list is the i->j pair
        # invmap[i] tells you which Wyckoff position i maps to (in the sitelist)
        # trying to avoid under-/over-flow
        siteene = np.array([ betaene[w] for w in self.invmap])
        sitepre = np.array([ pre[w] for w in self.invmap])
        return [ [ pT*np.exp(siteene[i]-beT)/sitepre[i]
                   for (i,j), dx in t ]
                 for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT) ]

    def symmratelist(self, pre, betaene, preT, betaeneT):
        """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
        # the ij tuple in each transition list is the i->j pair
        # invmap[i] tells you which Wyckoff position i maps to (in the sitelist)
        # trying to avoid under-/over-flow
        siteene = np.array([ betaene[w] for w in self.invmap])
        sitepre = np.array([ pre[w] for w in self.invmap])
        return [ [ pT*np.exp(0.5*siteene[i]+0.5*siteene[j]-beT)/np.sqrt(sitepre[i]*sitepre[j])
                   for (i,j), dx in t ]
                 for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT) ]

    def siteDipoles(self, dipoles):
        """
        Returns a list of the elastic dipole on each site, given the dipoles
        for the representatives
        :param dipoles: list of dipoles for the first representative site
        :return: array of dipole for each site [site][3][3]
        """
        # difficult to do with list comprehension since we're mapping from Wyckoff positions
        # to site indices; need to create the "blank" list first, then map into it.
        lis = np.zeros((self.N, 3, 3)) # blank list to index into
        for dipole, basis, sites, groupops in zip(dipoles, self.siteSymmTensorBasis,
                                                  self.sitelist, self.sitegroupops):
            symmdipole = crystal.ProjectTensorBasis(dipole, basis)
            for i, g in zip(sites, groupops):
                lis[i] = self.crys.g_tensor(g, symmdipole)
        return lis
        # return [ dipoles[w] for i,w in enumerate(self.invmap) ]

    def jumpDipoles(self, dipoles):
        """
        Returns a list of the elastic dipole for each transition, given the dipoles
        for the representatives
        :param dipoles: list of dipoles for the first representative transition
        :return: lists of lists of dipole for each transition
        """
        # symmetrize them first via projection
        symmdipoles = [ crystal.ProjectTensorBasis(dipole, basis)
                        for dipole, basis in zip(dipoles, self.jumpSymmTensorBasis)]
        return [ [ self.crys.g_tensor(g, dipole) for g in groupops ]
                 for groupops, dipole in zip(self.jumpgroupops, symmdipoles) ]

    def diffusivity(self, pre, betaene, preT, betaeneT, CalcDeriv = False):
        """
        Computes the diffusivity for our element given prefactors and energies/kB T.
        Also returns the negative derivative of diffusivity with respect to beta (used to compute
        the activation barrier tensor) if CalcDeriv = True
        The input list order corresponds to the sitelist and jumpnetwork

        Parameters
        ----------
        pre : list of prefactors for unique sites
        betaene : list of site energies divided by kB T
        preT : list of prefactors for transition states
        betaeneT: list of transition state energies divided by kB T

        Returns
        -------
        if CalcDeriv:
          D[3,3], DE[3,3] : diffusivity as 3x3 tensor, diffusivity times activation barrier
        else:
          D[3,3] : diffusivity as 3x3 tensor
        """
        if __debug__:
            if len(pre) != len(self.sitelist): raise IndexError("length of prefactor {} doesn't match sitelist".format(pre))
            if len(betaene) != len(self.sitelist): raise IndexError("length of energies {} doesn't match sitelist".format(betaene))
            if len(preT) != len(self.jumpnetwork): raise IndexError("length of prefactor {} doesn't match jump network".format(preT))
            if len(betaeneT) != len(self.jumpnetwork): raise IndexError("length of energies {} doesn't match jump network".format(betaeneT))
        rho = self.siteprob(pre, betaene)
        sqrtrho = np.sqrt(rho)
        ratelist = self.ratelist(pre, betaene, preT, betaeneT)
        symmratelist = self.symmratelist(pre, betaene, preT, betaeneT)
        omega_ij = np.zeros((self.N, self.N))
        domega_ij = np.zeros((self.N, self.N))
        bias_i = np.zeros((self.N, 3))
        dbias_i = np.zeros((self.N, 3))
        D0 = np.zeros((3,3))
        Db = np.zeros((3,3))
        # bookkeeping for energies:
        siteene = np.array([ betaene[w] for w in self.invmap])
        # transene = [ [ bET for (i,j), dx in t ] for t, bET in zip(self.jumpnetwork, betaeneT)]
        Eave = np.dot(rho, siteene)

        for transitionset, rates, symmrates, bET in zip(self.jumpnetwork, ratelist, symmratelist, betaeneT):
            for ((i,j), dx), rate, symmrate in zip(transitionset, rates, symmrates):
                # symmrate = sqrtrho[i]*invsqrtrho[j]*rate
                omega_ij[i, j] += symmrate
                omega_ij[i, i] -= rate
                domega_ij[i, j] += symmrate*(bET - 0.5*(siteene[i]+siteene[j]))
                domega_ij[i, i] -= rate*(bET - siteene[i])
                bias_i[i] += sqrtrho[i]*rate*dx
                dbias_i[i] += sqrtrho[i]*rate*dx*(bET - 0.5*(siteene[i]+Eave))
                D0 += 0.5*np.outer(dx, dx)*rho[i]*rate
                Db += 0.5*np.outer(dx, dx)*rho[i]*rate*(bET - Eave)
        if self.NV > 0:
            # NOTE: there's probably a SUPER clever way to do this with higher dimensional arrays and dot...
            omega_v = np.zeros((self.NV, self.NV))
            domega_v = np.zeros((self.NV, self.NV))
            bias_v = np.zeros(self.NV)
            dbias_v = np.zeros(self.NV)
            for a, va in enumerate(self.VectorBasis):
                bias_v[a] = np.trace(np.dot(bias_i.T, va))
                dbias_v[a] = np.trace(np.dot(dbias_i.T, va))
                for b, vb in enumerate(self.VectorBasis):
                    omega_v[a,b] = np.trace(np.dot(va.T, np.dot(omega_ij, vb)))
                    domega_v[a,b] = np.trace(np.dot(va.T, np.dot(domega_ij, vb)))
            gamma_v = self.bias_solver(omega_v, bias_v)
            dgamma_v = np.dot(domega_v, gamma_v)
            D0 += np.dot(np.dot(self.VV, bias_v), gamma_v)
            Db += np.dot(np.dot(self.VV, dbias_v), gamma_v) \
                  + np.dot(np.dot(self.VV, gamma_v), dbias_v) \
                  - np.dot(np.dot(self.VV, gamma_v), dgamma_v)

        if CalcDeriv:
            return D0, Db
        else:
            return D0

    def elastodiffusion(self, pre, betaene, dipole, preT, betaeneT, dipoleT):
        """
        Computes the elastodiffusion tensor for our element given prefactors, energies/kB T,
        and elastic dipoles/kB T
        The input list order corresponds to the sitelist and jumpnetwork

        Parameters
        ----------
        pre : list of prefactors for unique sites
        betaene : list of site energies divided by kB T
        dipole: list of elastic dipoles divided by kB T
        preT : list of prefactors for transition states
        betaeneT: list of transition state energies divided by kB T
        dipoleT: list of elastic dipoles divided by kB T

        Returns
        -------
        D[3,3], dD[3,3,3,3] : diffusivity as 3x3 tensor and elastodiffusion tensor as 3x3x3x3 tensor
        """
        def vector_tensor_outer(v,a):
            """Construct the outer product of v and a"""
            va = np.zeros((3,3,3))
            for i,j,k in ((i,j,k) for i in range(3) for j in range(3) for k in range(3)):
                va[i,j,k] = v[i]*a[j,k]
            return va

        def tensor_tensor_outer(a, b):
            """Construct the outer product of a and b"""
            ab = np.zeros((3,3,3,3))
            for i,j,k,l in ((i,j,k,l) for i in range(3) for j in range(3) for k in range(3) for l in range(3)):
                ab[i,j,k,l] = a[i,j]*b[k,l]
            return ab

        if __debug__:
            if len(pre) != len(self.sitelist): raise IndexError("length of prefactor {} doesn't match sitelist".format(pre))
            if len(betaene) != len(self.sitelist): raise IndexError("length of energies {} doesn't match sitelist".format(betaene))
            if len(dipole) != len(self.sitelist): raise IndexError("length of dipoles {} doesn't match sitelist".format(dipole))
            if len(preT) != len(self.jumpnetwork): raise IndexError("length of prefactor {} doesn't match jump network".format(preT))
            if len(betaeneT) != len(self.jumpnetwork): raise IndexError("length of energies {} doesn't match jump network".format(betaeneT))
            if len(dipoleT) != len(self.jumpnetwork): raise IndexError("length of dipoles {} doesn't match jump network".format(dipoleT))
        rho = self.siteprob(pre, betaene)
        sqrtrho = np.sqrt(rho)
        ratelist = self.ratelist(pre, betaene, preT, betaeneT)
        symmratelist = self.symmratelist(pre, betaene, preT, betaeneT)
        omega_ij = np.zeros((self.N, self.N))
        bias_i = np.zeros((self.N, 3))
        biasP_i = np.zeros((self.N, 3,3,3))
        domega_ij = np.zeros((self.N, self.N, 3,3))
        sitedipoles = self.siteDipoles(dipole)
        jumpdipoles = self.jumpDipoles(dipoleT)
        dipoleave = np.tensordot(rho, sitedipoles, [(0), (0)]) # average dipole

        D0 = np.zeros((3,3))
        Dp = np.zeros((3,3,3,3))
        for transitionset, rates, symmrates, dipoles in zip(self.jumpnetwork, ratelist, symmratelist, jumpdipoles):
            for ((i,j), dx), rate, symmrate, dipole in zip(transitionset, rates, symmrates, dipoles):
                # symmrate = sqrtrho[i]*invsqrtrho[j]*rate
                omega_ij[i, j] += symmrate
                omega_ij[i, i] -= rate
                domega_ij[i, j] -= symmrate*(dipole - 0.5*(sitedipoles[i] + sitedipoles[j]))
                domega_ij[i, i] += rate*(dipole - sitedipoles[i])
                bias_i[i] += sqrtrho[i]*rate*dx
                biasP_i[i] += vector_tensor_outer(sqrtrho[i]*rate*dx, dipole - 0.5*(sitedipoles[i] + dipoleave))
                D0 += 0.5*np.outer(dx, dx)*rho[i]*rate
                Dp += 0.5*tensor_tensor_outer(np.outer(dx, dx)*rho[i]*rate, dipole - dipoleave)
        if self.NV > 0:
            omega_v = np.zeros((self.NV, self.NV))
            bias_v = np.zeros(self.NV)
            domega_v = np.zeros((self.NV, self.NV, 3,3))
            # NOTE: there's probably a SUPER clever way to do this with higher dimensional arrays and dot...
            for a, va in enumerate(self.VectorBasis):
                bias_v[a] = np.tensordot(bias_i, va, ((0,1),(0,1))) # can also use trace(dot(bias_i.T, va))
                for b, vb in enumerate(self.VectorBasis):
                    omega_v[a,b] = np.tensordot(va, np.tensordot(omega_ij, vb, ((1),(0))), ((0,1),(0,1)))
                    domega_v[a,b] = np.tensordot(va, np.tensordot(domega_ij, vb, ((1),(0))), ((0,1),(0,3)))
            gamma_v = self.bias_solver(omega_v, bias_v)
            dg = np.tensordot(domega_v, gamma_v,((1),(0)))
            # need to project gamma_v *back onto* our sites; not sure if we can just do with a dot since
            # self.VectorBasis is a list of Nx3 matrices
            gamma_i = sum( g*va for g, va in zip(gamma_v, self.VectorBasis) )
            D0 += np.dot(np.dot(self.VV, bias_v), gamma_v)
            for c,d in ((c,d) for c in range(3) for d in range(3)):
                Dp[:,:,c,d] += np.tensordot(gamma_i, biasP_i[:,:,c,d], ((0),(0))) + \
                               np.tensordot(biasP_i[:,:,c,d], gamma_i, ((0),(0)))
            Dp += np.tensordot(np.tensordot(self.VV, gamma_v, ((3),(0))), dg, ((2),(0)))

        for a,b,c,d in ((a,b,c,d) for a in range(3) for b in range(3) for c in range(3) for d in range(3)):
            if a==c:
                Dp[a,b,c,d] += 0.5*D0[b,d]
            if a==d:
                Dp[a,b,c,d] += 0.5*D0[b,c]
            if b==c:
                Dp[a,b,c,d] += 0.5*D0[a,d]
            if b==d:
                Dp[a,b,c,d] += 0.5*D0[a,c]
        return D0, Dp


# YAML tags
VACANCYTHERMOKINETICS_YAMLTAG = '!VacancyThermoKinetics'

class vacancyThermoKinetics(collections.namedtuple('vacancyThermoKinetics',
                                                   'pre betaene preT betaeneT')):
    """
    Class to store (in a hashable manner) the thermodynamics and kinetics for the vacancy
    :param pre: prefactors for sites
    :param betaene: energy for sites / kBT
    :param preT: prefactors for transition states
    :param betaeneT: transition state energy for sites / kBT
    """
    def __repr__(self):
        return "{}(pre={}, betaene={}, preT={}, betaeneT={})".format(self.__class__.__name__,
                                                                     self.pre, self.betaene,
                                                                     self.preT, self.betaeneT)

    def _asdict(self):
        """Return a proper dict"""
        return {'pre': self.pre, 'betaene': self.betaene, 'preT': self.preT, 'betaeneT': self.betaeneT}

    def __eq__(self, other):
        # Note: could scale all prefactors by min(pre) and subtract all energies by min(ene)...?
        return isinstance(other, self.__class__) and \
               np.allclose(self.pre, other.pre) and np.allclose(self.betaene, other.betaene) and \
               np.allclose(self.preT, other.preT) and np.allclose(self.betaeneT, other.betaeneT)

    def __ne__(self, other):
        return not __eq__(other)

    def __hash__(self):
        return hash(self.pre.data.tobytes() + self.betaene.data.tobytes() +
                    self.preT.data.tobytes() + self.betaeneT.data.tobytes())

    @staticmethod
    def vacancyThermoKinetics_representer(dumper, data):
        """Output a PairState"""
        # asdict() returns an OrderedDictionary, so pass through dict()
        # had to rewrite _asdict() for some reason...?
        return dumper.represent_mapping(VACANCYTHERMOKINETICS_YAMLTAG, data._asdict())

    @staticmethod
    def vacancyThermoKinetics_constructor(loader, node):
        """Construct a GroupOp from YAML"""
        # ** turns the dictionary into parameters for GroupOp constructor
        return vacancyThermoKinetics(**loader.construct_mapping(node, deep=True))

# HDF5 conversion routines: vTK indexed dictionaries
def vTKdict2arrays(vTKdict):
    """
    Takes a dictionary indexed by vTK objects, returns two arrays of vTK keys and values,
    and the splits to separate vTKarray back into vTK
    :param vTKdict: dictionary, indexed by vTK objects, whose entries are arrays
    :return vTKarray: array of vTK entries
    :return valarray: array of values
    :return vTKsplits: split placement for vTK entries
    """
    if len(vTKdict.keys()) == 0: return None, None, None
    vTKexample = [k for k in vTKdict.keys()][0]
    vTKsplits = np.cumsum(np.array([ len(v) for v in vTKexample ]))[:-1]
    vTKlist = []
    vallist = []
    for k, v in zip(vTKdict.keys(), vTKdict.values()):
        vTKlist.append(np.hstack(k)) # k.pre, k.betaene, k.preT, k.betaeneT
        vallist.append(v)
    return np.array(vTKlist), np.array(vallist), vTKsplits

def arrays2vTKdict(vTKarray, valarray, vTKsplits):
    """
    Takes two arrays of vTK keys and values, and the splits to separate vTKarray back into vTK
    and returns a dictionary indexed by the vTK.
    :param vTKarray: array of vTK entries
    :param valarray: array of values
    :param vTKsplits: split placement for vTK entries
    :return vTKdict: dictionary, indexed by vTK objects, whose entries are arrays
    """
    if all( x is None for x in (vTKarray, valarray, vTKsplits)): return {}
    vTKdict = {}
    for vTKa, val in zip(vTKarray, valarray):
        vTKdict[vacancyThermoKinetics(*np.hsplit(vTKa, vTKsplits))] = val
    return vTKdict

# database tags
SOLUTE_TAG = 's'
VACANCY_TAG = 'v'
SINGLE_DEFECT_TAG = '{type}:{u1:+06.3f},{u2:+06.3f},{u3:+06.3f}'
DOUBLE_DEFECT_TAG = '{state1}-{state2}'
OM0_TAG = 'omega0:{vac1}^{vac2}'
OM1_TAG = 'omega1:{solute}-{vac1}^{vac2}'
OM2_TAG = 'omega2:{complex1}^{complex2}'


class VacancyMediated(object):
    """
    A class to compute vacancy-mediated solute transport coefficients, specifically
    L_vv (vacancy diffusion), L_ss (solute), and L_sv (off-diagonal). As part of that,
    it determines *what* quantities are needed as inputs in order to perform this calculation.

    Based on crystal class. Also now includes its own GF calculator and cacheing.
    """
    def __init__(self, crys, chem, sitelist, jumpnetwork, Nthermo = 0):
        """
        Create our diffusion calculator for a given crystal structure, chemical identity,
        jumpnetwork (for the vacancy) and thermodynamic shell.

        :param crys: Crystal object
        :param chem: index identifying the diffusing species
        :param sitelist: list, grouped into Wyckoff common positions, of unique sites
        :param jumpnetwork: list of unique transitions as lists of ((i,j), dx)
        """
        if all(x is None for x in (crys, chem, sitelist, jumpnetwork)): return  # blank object
        self.crys = crys
        self.chem = chem
        self.sitelist = copy.deepcopy(sitelist)
        self.jumpnetwork = copy.deepcopy(jumpnetwork)
        self.N = sum(len(w) for w in sitelist)
        self.invmap = np.zeros(self.N, dtype=int)
        for ind,w in enumerate(sitelist):
            for i in w:
                self.invmap[i] = ind
        self.om0_jn= copy.deepcopy(jumpnetwork)
        self.GFcalc = GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.om0_jn, 4) # Nmax?
        # do some initial setup:
        self.thermo = stars.StarSet(self.jumpnetwork, self.crys, self.chem, Nthermo)
        self.NNstar = stars.StarSet(self.jumpnetwork, self.crys, self.chem, 1)
        # self.kinetic = self.thermo + self.NNstar
        self.vkinetic = stars.VectorStarSet()
        self.generate(Nthermo)
        # self.generatetags()  # vacancy, solute, solute-vacancy; omega0, omega1, omega2

    def generate(self, Nthermo):
        """
        Generate the necessary stars, vector-stars, and jump networks based on the thermodynamic range.
        :param Nthermo : range of thermodynamic interactions, in terms of "shells",
            which is multiple summations of jumpvect
        """
        if Nthermo == getattr(self, 'Nthermo', 0): return
        self.Nthermo = Nthermo

        self.thermo.generate(Nthermo)
        self.kinetic = self.thermo + self.NNstar
        self.vkinetic.generate(self.kinetic)
        # some indexing helpers:
        # thermo2kin maps star index in thermo to kinetic (should just be range(n), but we use this for safety)
        # kin2vacancy maps star index in kinetic to non-solute configuration from sitelist
        # outerkin is the list of stars that are in kinetic, but not in thermo
        # vstar2kin maps each vector star back to the corresponding star index
        # kin2vstar provides a list of vector stars indices corresponding to the same star index
        self.thermo2kin = [self.kinetic.starindex(self.thermo.states[s[0]]) for s in self.thermo.stars]
        self.kin2vacancy = [self.invmap[self.kinetic.states[s[0]].j] for s in self.kinetic.stars]
        self.outerkin = [s for s in range(self.kinetic.Nstars)
                         if self.thermo.stateindex(self.kinetic.states[self.kinetic.stars[s][0]]) is None]
        self.vstar2kin = [self.kinetic.index[Rs[0]] for Rs in self.vkinetic.vecpos]
        self.kin2vstar = [ [j for j in range(self.vkinetic.Nvstars) if self.vstar2kin[j] == i]
                           for i in range(self.kinetic.Nstars)]
        # jumpnetwork, jumptype (omega0), star-pair for jump
        self.om1_jn, self.om1_jt, self.om1_SP = self.kinetic.jumpnetwork_omega1()
        self.om2_jn, self.om2_jt, self.om2_SP = self.kinetic.jumpnetwork_omega2()
        # Prune the om1 list: remove entries that have jumps between stars in outerkin:
        # work in reverse order so that popping is safe (and most of the offending entries are at the end
        for i, SP in zip(reversed(range(len(self.om1_SP))), reversed(self.om1_SP)):
            if SP[0] in self.outerkin and SP[1] in self.outerkin:
                self.om1_jn.pop(i), self.om1_jt.pop(i), self.om1_SP.pop(i)
        # TODO: check the GF calculator against the range in GFstarset to make sure its adequate
        # Vector star set, generates a LOT of our calculation:
        self.GFexpansion, self.GFstarset = self.vkinetic.GFexpansion()
        # empty dictionaries to store GF values
        self.GFvalues = {}
        self.Lvvvalues = {}
        self.om1_om0, self.om1_om0escape, self.om1expansion, self.om1escape = \
            self.vkinetic.rateexpansions(self.om1_jn, self.om1_jt)
        # technically, we don't need om2_om0 for anything
        self.om2_om0, self.om2_om0escape, self.om2expansion, self.om2escape = \
            self.vkinetic.rateexpansions(self.om2_jn, self.om2_jt)
        self.om1_b0, self.om1bias = self.vkinetic.biasexpansions(self.om1_jn, self.om1_jt)
        self.om2_b0, self.om2bias = self.vkinetic.biasexpansions(self.om2_jn, self.om2_jt)
        # more indexing helpers:
        # kineticsvWyckoff: Wyckoff position of solute and vacancy for kinetic stars
        # omega0vacancyWyckoff: Wyckoff positions of initial and final position in omega0 jumps
        # omega1svsvWyckoff: Wyckoff positions of solute+vacancy(initial), solute+vacancy(final) for omega1
        # omega2svsvWyckoff: Wyckoff positions of solute+vacancy(initial), solute+vacancy(final) for omega2
        self.kineticsvWyckoff = [(self.invmap[PS.i], self.invmap[PS.j]) for PS in
                         [self.kinetic.states[si[0]] for si in self.kinetic.stars]]
        self.omega0vacancyWyckoff = [(self.invmap[jumplist[0][0][0]], self.invmap[jumplist[0][0][1]])
                                    for jumplist in self.om0_jn]
        self.omega1svsvWyckoff = [(self.invmap[self.kinetic.states[jumplist[0][0][0]].i],
                                   self.invmap[self.kinetic.states[jumplist[0][0][0]].j],
                                   self.invmap[self.kinetic.states[jumplist[0][0][1]].i],
                                   self.invmap[self.kinetic.states[jumplist[0][0][1]].j])
                                  for jumplist in self.om1_jn]
        self.omega2svsvWyckoff = [(self.invmap[self.kinetic.states[jumplist[0][0][0]].i],
                                   self.invmap[self.kinetic.states[jumplist[0][0][0]].j],
                                   self.invmap[self.kinetic.states[jumplist[0][0][1]].i],
                                   self.invmap[self.kinetic.states[jumplist[0][0][1]].j])
                                  for jumplist in self.om1_jn]

    # this is part of our *class* definition: list of data that can be directly assigned / read
    __HDF5list__ = ('chem', 'N', 'invmap', 'thermo2kin', 'kin2vacancy', 'outerkin', 'vstar2kin',
                    'om1_jt', 'om1_SP', 'om2_jt', 'om2_SP',
                    'GFexpansion',
                    'om1_om0', 'om1_om0escape', 'om1expansion', 'om1escape',
                    'om2_om0', 'om2_om0escape', 'om2expansion', 'om2escape',
                    'om1_b0', 'om1bias', 'om2_b0', 'om2bias',
                    'kineticsvWyckoff', 'omega0vacancyWyckoff', 'omega1svsvWyckoff',
                    'omega2svsvWyckoff')

    def addhdf5(self, HDF5group):
        """
        Adds an HDF5 representation of object into an HDF5group (needs to already exist).

        Example: if f is an open HDF5, then StarSet.addhdf5(f.create_group('StarSet')) will
          (1) create the group named 'StarSet', and then (2) put the StarSet representation in that group.
        :param HDF5group: HDF5 group
        """
        HDF5group.attrs['type'] = self.__class__.__name__
        HDF5group['crystal_yaml'] = crystal.yaml.dump(self.crys)
        HDF5group['crystal_yaml'].attrs['pythonrep'] = self.crys.__repr__()
        HDF5group['crystal_lattice'] = self.crys.lattice.T
        basislist, basisindex = stars.doublelist2flatlistindex(self.crys.basis)
        HDF5group['crystal_basisarray'], HDF5group['crystal_basisindex'] = \
            np.array(basislist), basisindex
        # a long way around, but if you want to store an array of variable length strings, this is how to do it:
        # import h5py
        # HDF5group.create_dataset('crystal_chemistry', data=np.array(self.crys.chemistry, dtype=object),
        #                          dtype=h5py.special_dtype(vlen=str))
        HDF5group['crystal_chemistry'] = np.array(self.crys.chemistry, dtype='S')
        # arrays that we can deal with:
        for internal in self.__HDF5list__:
            HDF5group[internal] = getattr(self, internal)
        # convert jumplist:
        jumplist, jumpindex = stars.doublelist2flatlistindex(self.jumpnetwork)
        HDF5group['jump_ij'], HDF5group['jump_dx'], HDF5group['jump_index'] = \
            np.array([np.array((i,j)) for ((i,j), dx) in jumplist]), \
            np.array([dx for ((i,j), dx) in jumplist]), \
            jumpindex
        # objects with their own addhdf5 functionality:
        self.GFcalc.addhdf5(HDF5group.create_group('GFcalc'))
        self.thermo.addhdf5(HDF5group.create_group('thermo'))
        self.NNstar.addhdf5(HDF5group.create_group('NNstar'))
        self.kinetic.addhdf5(HDF5group.create_group('kinetic'))
        self.vkinetic.addhdf5(HDF5group.create_group('vkinetic'))
        self.GFstarset.addhdf5(HDF5group.create_group('GFstarset'))

        # jump networks:
        jumplist, jumpindex = stars.doublelist2flatlistindex(self.om1_jn)
        HDF5group['omega1_ij'], HDF5group['omega1_dx'], HDF5group['omega1_index'] = \
            np.array([np.array((i,j)) for ((i,j), dx) in jumplist]), \
            np.array([dx for ((i,j), dx) in jumplist]), \
            jumpindex

        jumplist, jumpindex = stars.doublelist2flatlistindex(self.om2_jn)
        HDF5group['omega2_ij'], HDF5group['omega2_dx'], HDF5group['omega2_index'] = \
            np.array([np.array((i,j)) for ((i,j), dx) in jumplist]), \
            np.array([dx for ((i,j), dx) in jumplist]), \
            jumpindex

        HDF5group['kin2vstar_array'], HDF5group['kin2vstar_index'] = \
            stars.doublelist2flatlistindex(self.kin2vstar)

        if self.GFvalues != {}:
            HDF5group['GFvalues_vTK'], HDF5group['GFvalues_values'], HDF5group['GFvalues_splits'] = \
                vTKdict2arrays(self.GFvalues)
            HDF5group['Lvvvalues_vTK'], HDF5group['Lvvvalues_values'], HDF5group['Lvvvalues_splits'] = \
                vTKdict2arrays(self.Lvvvalues)

    @classmethod
    def loadhdf5(cls, HDF5group):
        """
        Creates a new VacancyMediated diffuser from an HDF5 group.
        :param HDFgroup: HDF5 group
        :return: new StarSet object
        """
        diffuser = cls(None, None, None, None)  # initialize
        diffuser.crys = crystal.yaml.load(HDF5group['crystal_yaml'].value)
        for internal in cls.__HDF5list__:
            setattr(diffuser, internal, HDF5group[internal].value)
        diffuser.sitelist = [[] for i in range(max(diffuser.invmap)+1)]
        for i, site in enumerate(diffuser.invmap):
            diffuser.sitelist[site].append(i)

        # convert jumplist:
        diffuser.jumpnetwork = stars.flatlistindex2doublelist([((ij[0], ij[1]), dx) for ij, dx in \
                                                               zip(HDF5group['jump_ij'].value, HDF5group['jump_dx'].value)], HDF5group['jump_index'])
        diffuser.om0_jn= copy.deepcopy(diffuser.jumpnetwork)

        # objects with their own addhdf5 functionality:
        diffuser.GFcalc = GFcalc.GFCrystalcalc.loadhdf5(diffuser.crys, HDF5group['GFcalc'])
        diffuser.thermo = stars.StarSet.loadhdf5(diffuser.crys, HDF5group['thermo'])
        diffuser.NNstar = stars.StarSet.loadhdf5(diffuser.crys, HDF5group['NNstar'])
        diffuser.kinetic = stars.StarSet.loadhdf5(diffuser.crys, HDF5group['kinetic'])
        diffuser.vkinetic = stars.VectorStarSet.loadhdf5(diffuser.kinetic, HDF5group['vkinetic'])
        diffuser.GFstarset = stars.StarSet.loadhdf5(diffuser.crys, HDF5group['GFstarset'])

        # jump networks:
        diffuser.om1_jn = stars.flatlistindex2doublelist([((ij[0], ij[1]), dx) for ij, dx in \
                                                          zip(HDF5group['omega1_ij'].value, HDF5group['omega1_dx'].value)], HDF5group['omega1_index'])
        diffuser.om2_jn = stars.flatlistindex2doublelist([((ij[0], ij[1]), dx) for ij, dx in \
                                                          zip(HDF5group['omega2_ij'].value, HDF5group['omega2_dx'].value)], HDF5group['omega2_index'])

        diffuser.kin2vstar = stars.flatlistindex2doublelist(HDF5group['kin2vstar_array'],
                                                            HDF5group['kin2vstar_index'])
        if 'GFvalues_vTK' in HDF5group:
            diffuser.GFvalues = arrays2vTKdict(HDF5group['GFvalues_vTK'],
                                               HDF5group['GFvalues_values'],
                                               HDF5group['GFvalues_splits'])
            diffuser.Lvvvalues = arrays2vTKdict(HDF5group['Lvvvalues_vTK'],
                                                HDF5group['Lvvvalues_values'],
                                                HDF5group['Lvvvalues_splits'])
        else:
            diffuser.GFvalues, diffuser.Lvvvalues = {}, {}

        return diffuser

    def interactlist(self):
        """
        Return a list of solute-vacancy configurations for interactions. The points correspond
        to a vector between a solute atom and a vacancy. Defined by Stars.

        :return statelist: list of PairStates for the solute-vacancy interactions
        """
        if 0 == getattr(self, 'Nthermo', 0): raise ValueError('Need to set thermodynamic range first')
        return [self.thermo.states[s[0]] for s in self.thermo.stars]

    def omegalist(self, fivefreqindex=1):
        """
        Return a list of pairs of endpoints for a vacancy jump, corresponding to omega1 or omega2
        Solute at the origin, vacancy hopping between two sites. Defined by om1_jumpnetwork
        :param fivefreqindex: 1 or 2, corresponding to omega1 or omega2

        :return omegalist: list of tuples of PairStates
        :return omegajumptype: index of corresponding omega0 jumptype
        """
        # TODO: would be useful to come up with a "minimal" set of states to use here
        if 0 == getattr(self, 'Nthermo', 0): raise ValueError('Need to set thermodynamic range first')
        om, jt = {1: (self.om1_jn, self.om1_jt),
                  2: (self.om2_jn, self.om2_jt)}.get(fivefreqindex, (None,None))
        if om is None: raise ValueError('Five frequency index should be 1 or 2')
        return [(self.kinetic.states[jlist[0][0][0]], self.kinetic.states[jlist[0][0][1]]) for jlist in om], \
               jt.copy()

    def maketracerpreene(self, preT0, eneT0, **ignoredextraarguments):
        """
        Generates corresponding energies / prefactors for an isotopic tracer. Returns a dictionary.
        (we ignore extra arguments so that a dictionary including additional entries can be passed)

        :param preT0[Nomeg0]: prefactor for vacancy jump transitions (follows jumpnetwork)
        :param eneT0[Nomega0]: transition energy state for vacancy jumps

        :return preS[NWyckoff]: prefactor for solute formation
        :return eneS[NWyckoff]: solute formation energy
        :return preSV[Nthermo]: prefactor for solute-vacancy interaction
        :return eneSV[Nthermo]: solute-vacancy binding energy
        :return preT1[Nomega1]: prefactor for omega1-style transitions (follows om1_jn)
        :return eneT1[Nomega1]: transition energy for omega1-style jumps
        :return preT2[Nomega2]: prefactor for omega2-style transitions (follows om2_jn)
        :return eneT2[Nomega2]: transition energy for omega2-style jumps
        """
        preS = np.ones(len(self.sitelist))
        eneS = np.zeros(len(self.sitelist))
        preSV = np.ones(self.thermo.Nstars)
        eneSV = np.zeros(self.thermo.Nstars)
        preT1 = np.ones(len(self.om1_jn))
        eneT1 = np.zeros(len(self.om1_jn))
        for j, jt in zip(itertools.count(), self.om1_jt): preT1[j], eneT1[j] = preT0[jt], eneT0[jt]
        preT2 = np.ones(len(self.om2_jn))
        eneT2 = np.zeros(len(self.om2_jn))
        for j, jt in zip(itertools.count(), self.om2_jt): preT2[j], eneT2[j] = preT0[jt], eneT0[jt]
        return {'preS': preS, 'eneS': eneS, 'preSV': preSV, 'eneSV': eneSV,
                'preT1': preT1, 'eneT1': eneT1, 'preT2': preT2, 'eneT2': eneT2}

    def makeLIMBpreene(self, preS, eneS, preSV, eneSV, preT0, eneT0, **ignoredextraarguments):
        """
        Generates corresponding energies / prefactors for corresponding to LIMB
        (Linearized interpolation of migration barrier approximation). Returns a dictionary.
        (we ignore extra arguments so that a dictionary including additional entries can be passed)

        :param preS[NWyckoff]: prefactor for solute formation
        :param eneS[NWyckoff]: solute formation energy
        :param preSV[Nthermo]: prefactor for solute-vacancy interaction
        :param eneSV[Nthermo]: solute-vacancy binding energy
        :param preT0[Nomeg0]: prefactor for vacancy jump transitions (follows jumpnetwork)
        :param eneT0[Nomega0]: transition energy for vacancy jumps

        :return preT1[Nomega1]: prefactor for omega1-style transitions (follows om1_jn)
        :return eneT1[Nomega1]: transition energy/kBT for omega1-style jumps
        :return preT2[Nomega2]: prefactor for omega2-style transitions (follows om2_jn)
        :return eneT2[Nomega2]: transition energy/kBT for omega2-style jumps
        """
        # we need the prefactors and energies for all of our kinetic stars... without the
        # vacancy part (since that reference is already in preT0 and eneT0); we're going
        # to add these to preT0 and eneT0 to get the TS prefactor/energy for w1 and w2 jumps
        eneSVkin = np.array([eneS[s] for (s,v) in self.kineticsvWyckoff])
        preSVkin = np.array([preS[s] for (s,v) in self.kineticsvWyckoff])
        for tindex, kindex in enumerate(self.thermo2kin):
            eneSVkin[kindex] += eneSV[tindex]
            preSVkin[kindex] *= preSV[tindex]
        preT1 = np.ones(len(self.om1_jn))
        eneT1 = np.zeros(len(self.om1_jn))
        for j, jt, SP in zip(itertools.count(), self.om1_jt, self.om1_SP):
            # need to include solute energy / prefactors
            preT1[j] = preT0[jt]*np.sqrt(preSVkin[SP[0]]*preSVkin[SP[1]])
            eneT1[j] = eneT0[jt] + 0.5*(eneSVkin[SP[0]]+eneSVkin[SP[1]])
        preT2 = np.ones(len(self.om2_jn))
        eneT2 = np.zeros(len(self.om2_jn))
        for j, jt, SP in zip(itertools.count(), self.om2_jt, self.om2_SP):
            # need to include solute energy / prefactors
            preT2[j] = preT0[jt]*np.sqrt(preSVkin[SP[0]]*preSVkin[SP[1]])
            eneT2[j] = eneT0[jt] + 0.5*(eneSVkin[SP[0]]+eneSVkin[SP[1]])
        return {'preT1': preT1, 'eneT1': eneT1, 'preT2': preT2, 'eneT2': eneT2}

    @staticmethod
    def preene2betafree(kT, preV, eneV, preS, eneS, preSV, eneSV,
                        preT0, eneT0, preT1, eneT1, preT2, eneT2, **ignoredextraarguments):
        """
        Read in a series of prefactors (e^(S/kB)) and energies, and return beta*free energy for
        energies and transition state energies. Used to provide scaled values to Lij() and _lij().
        Can specify all of the entries using a dictionary; e.g., preene2betafree(kT, **data_dict)
        and then send that output as input to Lij: Lij(*preene2betafree(kT, **data_dict))
        (we ignore extra arguments so that a dictionary including additional entries can be passed)

        :param kT: temperature times Boltzmann's constant kB
        :param preV: prefactor for vacancy formation (prod of inverse vibrational frequencies)
        :param eneV: vacancy formation energy
        :param preS: prefactor for solute formation (prod of inverse vibrational frequencies)
        :param eneS: solute formation energy
        :param preSV: excess prefactor for solute-vacancy binding
        :param eneSV: solute-vacancy binding energy
        :param preT0: prefactor for vacancy transition state
        :param eneT0: energy for vacancy transition state (relative to eneV)
        :param preT1: prefactor for vacancy swing transition state
        :param eneT1: energy for vacancy swing transition state (relative to eneV + eneS + eneSV)
        :param preT2: prefactor for vacancy exchange transition state
        :param eneT2: energy for vacancy exchange transition state (relative to eneV + eneS + eneSV)

        :return bFV: beta*eneV - ln(preV) (relative to minimum value)
        :return bFS: beta*eneS - ln(preS) (relative to minimum value)
        :return bFSV: beta*eneSV - ln(preSV) (excess)
        :return bFT0: beta*eneT0 - ln(preT0) (relative to minimum value of bFV)
        :return bFT1: beta*eneT1 - ln(preT1) (relative to minimum value of bFV + bFS)
        :return bFT2: beta*eneT2 - ln(preT2) (relative to minimum value of bFV + bFS)
        """
        # do anything to treat kT -> 0?
        beta = 1/kT
        bFV = beta*eneV - np.log(preV)
        bFS = beta*eneS - np.log(preS)
        bFSV = beta*eneSV - np.log(preSV)
        bFT0 = beta*eneT0 - np.log(preT0)
        bFT1 = beta*eneT1 - np.log(preT1)
        bFT2 = beta*eneT2 - np.log(preT2)

        bFVmin = np.min(bFV)
        bFSmin = np.min(bFS)
        bFV -= bFVmin
        bFS -= bFSmin
        bFT0 -= bFVmin
        bFT1 -= bFVmin + bFSmin
        bFT2 -= bFVmin + bFSmin
        return bFV, bFS, bFSV, bFT0, bFT1, bFT2

    def _symmetricandescaperates(self, bFV, bFS, bFSV, bFT0, bFT1, bFT2):
        """
        Compute the symmetric, escape, and escape reference rates. Used by _lij().

        :param bFV[NWyckoff]: beta*eneV - ln(preV) (relative to minimum value)
        :param bFS[NWyckoff]: beta*eneS - ln(preS) (relative to minimum value)
        :param bFSV[Nthermo]: beta*eneSV - ln(preSV) (excess)
        :param bFT0[Nomega0]: beta*eneT0 - ln(preT0) (relative to minimum value of bFV)
        :param bFT1[Nomega1]: beta*eneT1 - ln(preT1) (relative to minimum value of bFV + bFS)
        :param bFT2[Nomega2]: beta*eneT2 - ln(preT2) (relative to minimum value of bFV + bFS)

        :return omega0[Nomega0]: symmetric rate for omega0 jumps
        :return omega1[Nomega1]: symmetric rate for omega1 jumps
        :return omega2[Nomega2]: symmetric rate for omega2 jumps
        :return omega0escape[NWyckoff, Nomega0]: escape rate elements for omega0 jumps
        :return omega1escape[NVstars, Nomega1]: escape rate elements for omega1 jumps
        :return omega2escape[NVstars, Nomega2]: escape rate elements for omega2 jumps
        :return omega1_om0escape[NVstars, Nomega0]: reference escape rate elements for omega1 jumps
        :return omega2_om0escape[NVstars, Nomega0]: reference escape rate elements for omega1 jumps
        """
        # omega0 = np.array([np.exp(0.5*(bFV[self.invmap[jump[0][0][0]]] + bFV[self.invmap[jump[0][0][1]]])-bF)
        #                    for bF, jump in zip(bFT0, self.om0_jn)])
        omega0 = np.zeros(len(self.om0_jn))
        omega0escape = np.zeros((len(self.sitelist), len(self.om0_jn)))
        for j, bF, (v1,v2) in zip(itertools.count(), bFT0, self.omega0vacancyWyckoff):
            omega0escape[v1,j] = np.exp(-bF + bFV[v1])
            omega0escape[v2,j] = np.exp(-bF + bFV[v2])
            omega0[j] = np.sqrt(omega0escape[v1,j]*omega0escape[v2,j])
        omega1 = np.zeros(len(self.om1_jn))
        omega1escape = np.zeros((self.vkinetic.Nvstars, len(self.om1_jn)))
        omega1_om0escape = np.zeros((self.vkinetic.Nvstars, len(self.om0_jn)))
        for j, (s1,v1,s2,v2), jumptype, (st1, st2), bFT in zip(itertools.count(), self.omega1svsvWyckoff,
                                                               self.om1_jt, self.om1_SP, bFT1):
            # print(s1, v1, st1, s2, v2, st2)
            # print(bFS[s1],bFV[v1],bFSV[st1],bFS[s2],bFV[v2],bFSV[st2])
            omF, omB = np.exp(-bFT+bFS[s1]+bFV[v1]+bFSV[st1]), np.exp(-bFT+bFS[s2]+bFV[v2]+bFSV[st2])
            omega1[j] = np.sqrt(omF*omB)
            for vst1 in self.kin2vstar[st1]:
                omega1escape[vst1, j],omega1_om0escape[vst1, jumptype] = omF,omega0escape[v1, jumptype]
            for vst2 in self.kin2vstar[st2]:
                omega1escape[vst2, j],omega1_om0escape[vst2, jumptype] = omB,omega0escape[v2, jumptype]
        omega2 = np.zeros(len(self.om2_jn))
        omega2escape = np.zeros((self.vkinetic.Nvstars, len(self.om2_jn)))
        omega2_om0escape = np.zeros((self.vkinetic.Nvstars, len(self.om0_jn)))
        for j, (s1,v1,s2,v2), jumptype, (st1, st2), bFT in zip(itertools.count(), self.omega2svsvWyckoff,
                                                               self.om2_jt, self.om2_SP, bFT2):
            omF, omB = np.exp(-bFT+bFS[s1]+bFV[v1]+bFSV[st1]), np.exp(-bFT+bFS[s2]+bFV[v2]+bFSV[st2])
            omega2[j] = np.sqrt(omF*omB)
            for vst1 in self.kin2vstar[st1]:
                omega2escape[vst1, j],omega2_om0escape[vst1, jumptype] = omF,omega0escape[v1, jumptype]
            for vst2 in self.kin2vstar[st2]:
                omega2escape[vst2, j],omega2_om0escape[vst2, jumptype] = omB,omega0escape[v2, jumptype]
        return omega0, omega1, omega2, \
               omega0escape, omega1escape, omega2escape, \
               omega1_om0escape, omega2_om0escape

    def Lij(self, bFV, bFS, bFSV, bFT0, bFT1, bFT2):
        """
        Calculates the transport coefficients: Lvv, L0ss, L2ss, L1sv, L1vv from the scaled free energies.
        The Green function entries are calculated from the omega0 info. As this is the most
        time-consuming part of the calculation, we cache these values with a dictionary
        and hash function.
        Used by Lij.

        :param bFV[NWyckoff]: beta*eneV - ln(preV) (relative to minimum value)
        :param bFS[NWyckoff]: beta*eneS - ln(preS) (relative to minimum value)
        :param bFSV[Nthermo]: beta*eneSV - ln(preSV) (excess)
        :param bFT0[Nomega0]: beta*eneT0 - ln(preT0) (relative to minimum value of bFV)
        :param bFT1[Nomega1]: beta*eneT1 - ln(preT1) (relative to minimum value of bFV + bFS)
        :param bFT2[Nomega2]: beta*eneT2 - ln(preT2) (relative to minimum value of bFV + bFS)

        :return Lvv[3, 3]: vacancy-vacancy; needs to be multiplied by cv/kBT
        :return Lss[3, 3]: solute-solute; needs to be multiplied by cv*cs/kBT
        :return Lsv[3, 3]: solute-vacancy; needs to be multiplied by cv*cs/kBT
        :return Lvv1[3, 3]: vacancy-vacancy correction due to solute; needs to be multiplied by cv*cs/kBT
        """
        # 1. bare vacancy diffusivity and Green's function
        vTK = vacancyThermoKinetics(pre=np.ones_like(bFV), betaene=bFV,
                                    preT=np.ones_like(bFT0), betaeneT=bFT0)
        GF = self.GFvalues.get(vTK)
        L0vv = self.Lvvvalues.get(vTK)
        if GF is None:
            # calculate, and store in dictionary for cache:
            self.GFcalc.SetRates(**(vTK._asdict()))
            L0vv = self.GFcalc.Diffusivity()
            GF = np.array([self.GFcalc(PS.i, PS.j, PS.dx)
                           for PS in
                           [self.GFstarset.states[s[0]] for s in self.GFstarset.stars]])
            self.Lvvvalues[vTK] = L0vv
            self.GFvalues[vTK] = GF.copy()

        # 2. set up probabilities for solute-vacancy configurations
        probV = np.array([np.exp(min(bFV)-bFV[wi]) for wi in self.invmap])
        probV *= self.N/np.sum(probV)  # normalize
        probS = np.array([np.exp(min(bFS)-bFS[wi]) for wi in self.invmap])
        probS *= self.N/np.sum(probS)  # normalize
        bFSVkin = np.array([bFS[s]+bFV[v] for (s,v) in self.kineticsvWyckoff])
        prob = np.array([probS[s]*probV[v] for (s,v) in self.kineticsvWyckoff])
        for tindex, kindex in enumerate(self.thermo2kin):
            bFSVkin[kindex] += bFSV[tindex]
            prob[kindex] *= np.exp(-bFSV[tindex])

        # 3. set up symmetric rates: omega0, omega1, omega2
        #    and escape rates omega0escape, omega1escape, omega2escape,
        #    and reference escape rates omega1_om0escape, omega2_om0escape
        omega0, omega1, omega2, omega0escape, omega1escape, omega2escape, \
        omega1_om0escape, omega2_om0escape = \
            self._symmetricandescaperates(bFV, bFS, bFSVkin, bFT0, bFT1, bFT2)

        # 4. expand out: domega1, domega2, bias1, bias2
        # Note: we don't subtract off the equivalent of om1_om0 for omega2, because those
        # jumps correspond to the vacancy *landing* on the solute site, and those states are not included
        # however, the *escape* part must be referenced.
        delta_om = np.dot(self.om1expansion, omega1) - np.dot(self.om1_om0, omega0) + \
                   np.dot(self.om2expansion, omega2) # - np.dot(self.om2_om0, omega0)
        for sv in range(self.vkinetic.Nvstars):
            delta_om[sv,sv] += np.dot(self.om1escape[sv,:],omega1escape[sv,:]) - \
                               np.dot(self.om1_om0escape[sv,:], omega1_om0escape[sv,:]) + \
                               np.dot(self.om2escape[sv,:], omega2escape[sv,:]) - \
                               np.dot(self.om2_om0escape[sv,:], omega2_om0escape[sv,:])
        biasSvec = np.zeros(self.vkinetic.Nvstars)
        biasVvec = np.zeros(self.vkinetic.Nvstars)
        for sv,starindex in enumerate(self.vstar2kin):
            # note: our solute bias is negative of the contribution to the vacancy, and also the
            # reference value is 0
            svvacindex = self.kin2vacancy[starindex]  # vacancy
            biasSvec[sv] = -np.dot(self.om2bias[sv,:], omega2escape[sv,:])*np.sqrt(prob[starindex])
            biasVvec[sv] = np.dot(self.om1bias[sv,:], omega1escape[sv,:])*np.sqrt(prob[starindex]) - \
                           np.dot(self.om1_b0[sv,:], omega0escape[svvacindex,:])*np.sqrt(probV[svvacindex]) - \
                           biasSvec[sv] - \
                           np.dot(self.om2_b0[sv,:], omega0escape[svvacindex,:])*np.sqrt(probV[svvacindex])

        # 5. compute Onsager coefficients
        G0 = np.dot(self.GFexpansion, GF)
        G = np.dot(np.linalg.inv(np.eye(self.vkinetic.Nvstars) + np.dot(G0, delta_om)), G0)
        outer_etaVvec = np.dot(self.vkinetic.outer, np.dot(G, biasVvec))
        outer_etaSvec = np.dot(self.vkinetic.outer, np.dot(G, biasSvec))
        L2ss = np.dot(outer_etaSvec, biasSvec) /self.N
        L1sv = np.dot(outer_etaSvec, biasVvec) /self.N
        L1vv = np.dot(outer_etaVvec, biasVvec) /self.N
        # compute our bare solute diffusivity:
        L0ss = np.zeros((3,3))
        # Need to transpose omega2escape so that we're working with each escape, and then it
        # is indexed by which vector star it corresponds to.
        for om2, jumplist in zip(omega2escape.T, self.om2_jn):
            for (i,j), dx in jumplist:
                # index into kinetic stars to get the probability; to get the escape rate, have to index
                # back to the vector star (take the first entry since kin2vstar returns a list).
                L0ss += 0.5*np.outer(dx,dx) * om2[self.kin2vstar[self.kinetic.index[i]][0]] * \
                        prob[self.kinetic.index[i]]
        L0ss /= self.N
        return L0vv, L0ss + L2ss, -L0ss + L1sv, L1vv

crystal.yaml.add_representer(vacancyThermoKinetics, vacancyThermoKinetics.vacancyThermoKinetics_representer)
crystal.yaml.add_constructor(VACANCYTHERMOKINETICS_YAMLTAG, vacancyThermoKinetics.vacancyThermoKinetics_constructor)
