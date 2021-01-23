import numpy as np
import onsager.crystal as crystal
from states import *
from representations import *
from test_structs import *
# from gensets import *
import itertools
import unittest


class test_statemaking(unittest.TestCase):
    def setUp(self):
        famp0 = [np.array([1., 1., 0.]), np.array([1., 0., 0.])]
        famp12 = [np.array([1., 1., 1.]), np.array([1., 1., 0.])]
        self.family = [famp0, famp12]
        self.crys = tet2

    def test_dbStates(self):
        # check that symmetry analysis is correct
        dbstates = dbStates(self.crys, 0, self.family)
        self.assertEqual(len(dbstates.symorlist), 4)
        # check that every (i,or) set is accounted for
        sm = 0
        for i in dbstates.symorlist:
            sm += len(i)
        self.assertEqual(sm, len(dbstates.iorlist))

        # test indexmapping
        for gdumb in dbstates.G:
            # First check that all states are accounted for.
            self.assertEqual(len(gdumb.indexmap[0]), len(dbstates.iorlist))
            for idx1, tup1 in enumerate(dbstates.iorlist):
                i, o = tup1[0], tup1[1]
                R, (ch, inew) = dbstates.crys.g_pos(dbstates.G_crys[gdumb], np.array([0, 0, 0]), (dbstates.chem, i))
                onew = np.dot(gdumb.cartrot, o)
                count = 0
                for idx2, tup2 in enumerate(dbstates.iorlist):
                    if inew == tup2[0] and (np.allclose(tup2[1], onew, atol=dbstates.crys.threshold) or
                                      np.allclose(tup2[1], -onew, atol=dbstates.crys.threshold)):
                        count +=1
                        self.assertEqual(gdumb.indexmap[0][idx1], idx2, msg="{}, {}".format(gdumb.indexmap[0][idx1], idx2))
                self.assertEqual(count, 1)

        # test_indexedsymlist
        for i1, symindlist, symstatelist in zip(itertools.count(),dbstates.symIndlist, dbstates.symorlist):
            for stind, state in zip(symindlist, symstatelist):
                st_iorlist = dbstates.iorlist[stind]
                self.assertEqual(st_iorlist[0], state[0])
                self.assertTrue(np.allclose(st_iorlist[1], state[1], atol=dbstates.crys.threshold))

    # Test jumpnetwork
    def test_jnet0(self):
        # cube
        famp0 = [np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126]
        family = [famp0]
        pdbcontainer_cube = dbStates(cube, 0, family)
        jset_cube, jind_cube = pdbcontainer_cube.jumpnetwork(0.3, 0.01, 0.01)
        test_dbi = dumbbell(pdbcontainer_cube.getIndex((0, np.array([0.126, 0., 0.]))), np.array([0, 0, 0]))
        test_dbf = dumbbell(pdbcontainer_cube.getIndex((0, np.array([0.126, 0., 0.]))), np.array([0, 1, 0]))
        count = 0
        indices = []
        for i, jlist in enumerate(jset_cube):
            for q, j in enumerate(jlist):
                if j.state1 == test_dbi:
                    if j.state2 == test_dbf:
                        if j.c1 == j.c2 == -1:
                            count += 1
                            indices.append((i, q))
                            jtest = jlist
        # print (indices)
        self.assertEqual(count, 1)  # see that this jump has been taken only once into account
        self.assertEqual(len(jtest), 24)

        # Next FCC
        # test this out with FCC
        fcc = crystal.Crystal.FCC(0.55)
        famp0 = [np.array([1., 1., 0.]) / np.sqrt(2) * 0.2]
        family = [famp0]
        pdbcontainer_fcc = dbStates(fcc, 0, family)
        jset_fcc, jind_fcc = pdbcontainer_fcc.jumpnetwork(0.4, 0.01, 0.01)
        o1 = np.array([1., -1., 0.]) * 0.2 / np.sqrt(2)
        if any(np.allclose(-o1, o) for i, o in pdbcontainer_fcc.iorlist):
            o1 = -o1.copy()
        db1 = dumbbell(pdbcontainer_fcc.getIndex((0, o1)), np.array([0, 0, 0]))
        db2 = dumbbell(pdbcontainer_fcc.getIndex((0, o1)), np.array([0, 0, 1]))
        jmp = jump(db1, db2, 1, 1)
        jtest = []
        for jl in jset_fcc:
            for j in jl:
                if j == jmp:
                    jtest.append(jl)
        # see that the jump has been accounted just once.
        self.assertEqual(len(jtest), 1)
        # See that there 24 jumps. 24 0->0.
        self.assertEqual(len(jtest[0]), 24)

        # DC_Si - same symmetry as FCC, except twice the number of jumps, since we have two basis
        # atoms belonging to the same Wyckoff site, in a crystal with the same lattice vectors.
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 1., 0.]) / np.sqrt(2) * 0.2]
        family = [famp0]
        pdbcontainer_si = dbStates(DC_Si, 0, family)
        jset_si, jind_si = pdbcontainer_si.jumpnetwork(0.4, 0.01, 0.01)
        o1 = np.array([1., -1., 0.]) * 0.2 / np.sqrt(2)
        if any(np.allclose(-o1, o) for i, o in pdbcontainer_si.iorlist):
            o1 = -o1.copy()
        db1 = dumbbell(pdbcontainer_si.getIndex((0, o1)), np.array([0, 0, 0]))
        db2 = dumbbell(pdbcontainer_si.getIndex((0, o1)), np.array([0, 0, 1]))
        jmp = jump(db1, db2, 1, 1)
        jtest = []
        for jl in jset_si:
            for j in jl:
                if j == jmp:
                    jtest.append(jl)
        # see that the jump has been accounted just once.
        self.assertEqual(len(jtest), 1)
        # See that there 48 jumps. 24 0->0 and 24 1->1.
        self.assertEqual(len(jtest[0]), 48)

        # HCP
        Mg = crystal.Crystal.HCP(0.3294, chemistry=["Mg"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        pdbcontainer_hcp = dbStates(Mg, 0, family)
        jset_hcp, jind_hcp = pdbcontainer_hcp.jumpnetwork(0.45, 0.01, 0.01)
        o = np.array([0.145, 0., 0.])
        if any(np.allclose(-o, o1) for i, o1 in pdbcontainer_hcp.iorlist):
            o = -o + 0.
        db1 = dumbbell(pdbcontainer_hcp.getIndex((0, o)), np.array([0, 0, 0], dtype=int))
        db2 = dumbbell(pdbcontainer_hcp.getIndex((1, o)), np.array([0, 0, 0], dtype=int))
        testjump = jump(db1, db2, 1, 1)
        count = 0
        testlist = []
        for jl in jset_hcp:
            for j in jl:
                if j == testjump:
                    count += 1
                    testlist = jl
        self.assertEqual(len(testlist), 24)
        self.assertEqual(count, 1)

        # test_indices
        # First check if they have the same number of lists and elements
        jindlist = [jind_cube, jind_fcc, jind_si, jind_hcp]
        jsetlist = [jset_cube, jset_fcc, jset_si, jset_hcp]
        pdbcontainerlist = [pdbcontainer_cube, pdbcontainer_fcc, pdbcontainer_si, pdbcontainer_hcp]
        for pdbcontainer, jset, jind in zip(pdbcontainerlist, jsetlist, jindlist):
            self.assertEqual(len(jind), len(jset))
            # now check if all the elements are correctly correspondent
            for lindex in range(len(jind)):
                self.assertEqual(len(jind[lindex]), len(jset[lindex]))
                for jindex in range(len(jind[lindex])):
                    (i1, o1) = pdbcontainer.iorlist[jind[lindex][jindex][0][0]]
                    (i2, o2) = pdbcontainer.iorlist[jind[lindex][jindex][0][1]]
                    self.assertEqual(pdbcontainer.iorlist[jset[lindex][jindex].state1.iorind][0], i1)
                    self.assertEqual(pdbcontainer.iorlist[jset[lindex][jindex].state2.iorind][0], i2)
                    self.assertTrue(np.allclose(pdbcontainer.iorlist[jset[lindex][jindex].state1.iorind][1], o1))
                    self.assertTrue(np.allclose(pdbcontainer.iorlist[jset[lindex][jindex].state2.iorind][1], o2))
                    dx = disp(pdbcontainer, jset[lindex][jindex].state1, jset[lindex][jindex].state2)
                    self.assertTrue(np.allclose(dx, jind[lindex][jindex][1]))

    def test_mStates(self):
        dbstates = dbStates(self.crys, 0, self.family)
        mstates1 = mStates(self.crys, 0, self.family)

        # check that symmetry analysis is correct
        self.assertEqual(len(mstates1.symorlist), 4)

        # check that negative orientations are accounted for
        for i in range(4):
            self.assertEqual(len(mstates1.symorlist[i]) / len(dbstates.symorlist[i]), 2)

        # check that every (i,or) set is accounted for
        sm = 0
        for i in mstates1.symorlist:
            sm += len(i)
        self.assertEqual(sm, len(mstates1.iorlist))

        # check indexmapping
        for gdumb in mstates1.G:
            self.assertEqual(len(gdumb.indexmap[0]), len(mstates1.iorlist))
            for stateind, tup in enumerate(mstates1.iorlist):
                i, o = tup[0], tup[1]
                R, (ch, inew) = mstates1.crys.g_pos(mstates1.G_crys[gdumb], np.array([0, 0, 0]), (mstates1.chem, i))
                onew = np.dot(gdumb.cartrot, o)
                count = 0
                for j, t in enumerate(mstates1.iorlist):
                    if t[0] == inew and np.allclose(t[1], onew):
                        foundindex = j
                        count += 1
                self.assertEqual(count, 1)
                self.assertEqual(foundindex, gdumb.indexmap[0][stateind])

        # Check indexing of symlist
        for symind, symIndlist, symstlist in zip(itertools.count(), mstates1.symIndlist, mstates1.symorlist):
            for idx, state in zip(symIndlist, symstlist):
                self.assertEqual(mstates1.iorlist[idx][0],state[0])
                self.assertTrue(np.allclose(mstates1.iorlist[idx][1], state[1], atol=mstates1.crys.threshold))

    def test_mixedjumps(self):
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 1., 0.]) / np.sqrt(2) * 0.2]
        family = [famp0]
        mdbcontainer = mStates(DC_Si, 0, family)
        jset, jind = mdbcontainer.jumpnetwork(0.4, 0.01, 0.01)
        o1 = np.array([1., 1., 0.]) * 0.2 / np.sqrt(2)
        # if any(np.allclose(-o1, o) for i, o in pdbcontainer.iorlist):
        #     o1 = -o1.copy()
        # db1 = dumbbell(pdbcontainer_si.getIndex((0, o1)), np.array([0, 0, 0]))
        # db2 = dumbbell(pdbcontainer_si.getIndex((0, o1)), np.array([0, 0, 1]))
        test_dbi = dumbbell(mdbcontainer.getIndex((0, o1)), np.array([0, 0, 0]))
        test_dbf = dumbbell(mdbcontainer.getIndex((1, o1)), np.array([0, 0, 0]))
        count = 0
        jtest = None
        for i, jlist in enumerate(jset):
            for q, j in enumerate(jlist):
                if j.state1.db == test_dbi:
                    if j.state2.db == test_dbf:
                        if j.c1 == j.c2 == 1:
                            count += 1
                            jtest = jlist
        self.assertEqual(count, 1)  # see that this jump has been taken only once into account
        self.assertEqual(len(jtest), 48)

        # check if conditions for mixed dumbbell transitions are satisfied
        count = 0
        for jl in jset:
            for j in jl:
                if j.c1 == -1 or j.c2 == -1:
                    count += 1
                    break
                if not (j.state1.i_s == mdbcontainer.iorlist[j.state1.db.iorind][0] and
                        j.state2.i_s == mdbcontainer.iorlist[j.state2.db.iorind][0] and
                        np.allclose(j.state1.R_s, j.state1.db.R) and
                        np.allclose(j.state2.R_s, j.state2.db.R)):
                    count += 1
                    break
            if count == 1:
                break
        self.assertEqual(count, 0)

        # test_indices
        # First check if they have the same number of lists and elements
        self.assertEqual(len(jind), len(jset))
        # now check if all the elements are correctly correspondent
        for lindex in range(len(jind)):
            self.assertEqual(len(jind[lindex]), len(jset[lindex]))
            for jindex in range(len(jind[lindex])):
                (i1, o1) = mdbcontainer.iorlist[jind[lindex][jindex][0][0]]
                (i2, o2) = mdbcontainer.iorlist[jind[lindex][jindex][0][1]]
                self.assertEqual(mdbcontainer.iorlist[jset[lindex][jindex].state1.db.iorind][0], i1)
                self.assertEqual(mdbcontainer.iorlist[jset[lindex][jindex].state2.db.iorind][0], i2)
                self.assertTrue(np.allclose(mdbcontainer.iorlist[jset[lindex][jindex].state1.db.iorind][1], o1))
                self.assertTrue(np.allclose(mdbcontainer.iorlist[jset[lindex][jindex].state2.db.iorind][1], o2))

class test_2d(test_statemaking):
    def setUp(self):
        o = np.array([0.1, 0.])
        famp0 = [o.copy()]
        self.family = [famp0]

        latt = np.array([[1., 0.], [0., 1.]])
        self.crys = crystal.Crystal(latt, [np.array([0, 0])], ["A"])

    def test_dbStates(self):
        # check that symmetry analysis is correct
        dbstates = dbStates(self.crys, 0, self.family)
        self.assertEqual(len(dbstates.symorlist), 1)
        # check that every (i,or) set is accounted for
        sm = 0
        for i in dbstates.symorlist:
            sm += len(i)
        self.assertEqual(sm, len(dbstates.iorlist))

        # test indexmapping
        for gdumb in dbstates.G:
            # First check that all states are accounted for.
            self.assertEqual(len(gdumb.indexmap[0]), len(dbstates.iorlist))
            for idx1, tup1 in enumerate(dbstates.iorlist):
                i, o = tup1[0], tup1[1]
                R, (ch, inew) = dbstates.crys.g_pos(dbstates.G_crys[gdumb], np.array([0, 0]), (dbstates.chem, i))
                onew = np.dot(gdumb.cartrot, o)
                count = 0
                for idx2, tup2 in enumerate(dbstates.iorlist):
                    if inew == tup2[0] and (np.allclose(tup2[1], onew, atol=dbstates.crys.threshold) or
                                      np.allclose(tup2[1], -onew, atol=dbstates.crys.threshold)):
                        count +=1
                        self.assertEqual(gdumb.indexmap[0][idx1], idx2, msg="{}, {}".format(gdumb.indexmap[0][idx1], idx2))
                self.assertEqual(count, 1)

        # test_indexedsymlist
        for i1, symindlist, symstatelist in zip(itertools.count(), dbstates.symIndlist, dbstates.symorlist):
            for stind, state in zip(symindlist, symstatelist):
                st_iorlist = dbstates.iorlist[stind]
                self.assertEqual(st_iorlist[0], state[0])
                self.assertTrue(np.allclose(st_iorlist[1], state[1], atol=dbstates.crys.threshold))

    def test_mStates(self):
        dbstates = dbStates(self.crys, 0, self.family)
        mstates1 = mStates(self.crys, 0, self.family)

        # check that symmetry analysis is correct
        self.assertEqual(len(mstates1.symorlist), 1)

        # check that negative orientations are accounted for
        for i in range(len(mstates1.symorlist)):
            self.assertEqual(len(mstates1.symorlist[i]) / len(dbstates.symorlist[i]), 2)

        # check that every (i,or) set is accounted for
        sm = 0
        for i in mstates1.symorlist:
            sm += len(i)
        self.assertEqual(sm, len(mstates1.iorlist))

        # check indexmapping
        for gdumb in mstates1.G:
            self.assertEqual(len(gdumb.indexmap[0]), len(mstates1.iorlist))
            for stateind, tup in enumerate(mstates1.iorlist):
                i, o = tup[0], tup[1]
                R, (ch, inew) = mstates1.crys.g_pos(mstates1.G_crys[gdumb], np.array([0, 0]), (mstates1.chem, i))
                onew = np.dot(gdumb.cartrot, o)
                count = 0
                for j, t in enumerate(mstates1.iorlist):
                    if t[0] == inew and np.allclose(t[1], onew):
                        foundindex = j
                        count += 1
                self.assertEqual(count, 1)
                self.assertEqual(foundindex, gdumb.indexmap[0][stateind])

        # Check indexing of symlist
        for symind, symIndlist, symstlist in zip(itertools.count(), mstates1.symIndlist, mstates1.symorlist):
            for idx, state in zip(symIndlist, symstlist):
                self.assertEqual(mstates1.iorlist[idx][0],state[0])
                self.assertTrue(np.allclose(mstates1.iorlist[idx][1], state[1], atol=mstates1.crys.threshold))