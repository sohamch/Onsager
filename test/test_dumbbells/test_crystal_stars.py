import numpy as np
# from jumpnet3 import *
from onsager.crystalStars import *
from crysts import *
from onsager.crystal import DB_disp, DB_disp4, pureDBContainer, mixedDBContainer
from onsager.DB_structs import dumbbell, SdPair, jump, connector
# from gensets import *
import unittest
import collections


class test_StarSet(unittest.TestCase):

    def setUp(self):
        famp0 = [np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126]
        family = [famp0]
        self.pdbcontainer = pureDBContainer(cube, 0, family)
        self.mdbcontainer = mixedDBContainer(cube, 0, family)
        jset0 = self.pdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        jset2 = self.mdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        self.crys_stars = DBStarSet(self.pdbcontainer, self.mdbcontainer, jset0, jset2, 2)

    def test_generate(self):
        # test if the starset is generated correctly
        tot_st = 0
        for l in (self.crys_stars.stars[:self.crys_stars.mixedstartindex]):
            tot_st += len(l)
        self.assertEqual(tot_st, len(self.crys_stars.stateset))

        o = np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126
        dbInd = self.pdbcontainer.getIndex((0, o))
        db = dumbbell(dbInd, np.array([1, 0, 0], dtype=int))
        pair_test = SdPair(0, np.zeros(3, dtype=int), db)
        idxlist = []
        for idx, l in enumerate(self.crys_stars.stars):
            for state in l:
                if state == pair_test:
                    idxlist.append(idx)
        self.assertEqual(len(idxlist), 1)
        self.assertEqual(len(self.crys_stars.stars[idxlist[0]]), 6)

        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        pdbcontainer = pureDBContainer(DC_Si, 0, family)
        mdbcontainer = mixedDBContainer(DC_Si, 0, family)
        jset0 = pdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        jset2 = mdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        crys_stars = DBStarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)

        # check that starset is closed under symmetry
        for st in crys_stars.stateset:
            for gdumb in pdbcontainer.G:
                stnew, flipind = st.gop(pdbcontainer, gdumb)
                stnew -= stnew.R_s
                self.assertTrue(stnew in crys_stars.stateset)

        # Check that the stars are properly generated
        count_origin_states = 0
        for star in crys_stars.stars[:crys_stars.mixedstartindex]:
            repr = star[0]
            if repr.is_zero(crys_stars.pdbcontainer):
                count_origin_states += len(star)
            considered_already = set([])
            count = 0
            for gdumb in crys_stars.pdbcontainer.G:
                stnew = repr.gop(crys_stars.pdbcontainer, gdumb)[0]
                stnew -= stnew.R_s
                if stnew in star and not stnew in considered_already:
                    count += 1
                    considered_already.add(stnew)
            self.assertEqual(count, len(star))

        # Check that we have origin states
        self.assertTrue(count_origin_states, 6)

        # put in FCC test
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        FCC = crystal.Crystal(latt, [[np.array([0., 0., 0.])]], ["A"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        pdbcontainer = pureDBContainer(FCC, 0, family)
        mdbcontainer = mixedDBContainer(FCC, 0, family)
        jset0 = pdbcontainer.jumpnetwork(0.4, 0.01, 0.01)
        jset2 = mdbcontainer.jumpnetwork(0.4, 0.01, 0.01)
        crys_stars = DBStarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)

        # check that starset is closed under symmetry
        for st in crys_stars.stateset:
            for gdumb in pdbcontainer.G:
                stnew, flipind = st.gop(pdbcontainer, gdumb)
                stnew -= stnew.R_s
                self.assertTrue(stnew in crys_stars.stateset)
        # for fcc 1nn shell we can count explicitly - 3*12 = 36 + 3 origin states
        self.assertEqual(len(crys_stars.stateset), 39)

        # Now let's do a 1nn2 shell
        crys_stars = DBStarSet(pdbcontainer, mdbcontainer, jset0, jset2, 2)

        # check that starset is closed under symmetry
        for st in crys_stars.stateset:
            for gdumb in pdbcontainer.G:
                stnew, flipind = st.gop(pdbcontainer, gdumb)
                stnew -= stnew.R_s
                self.assertTrue(stnew in crys_stars.stateset)

        Rcounts = collections.defaultdict(int)
        for state in crys_stars.stateset:
            R = state.db.R
            Rcounts[(R[0], R[1], R[2])] += 1

        for (key, val) in Rcounts.items():
            self.assertEqual(val, 3)  # check that all three orientations are present at every R

    def test_indexing_stars(self):
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]

        hcp_Mg = crystal.Crystal.HCP(0.3294, chemistry=["Mg"])
        pdbcontainer1 = pureDBContainer(hcp_Mg, 0, family)
        mdbcontainer1 = mixedDBContainer(hcp_Mg, 0, family)
        jset0 = pdbcontainer1.jumpnetwork(0.45, 0.01, 0.01)
        jset2 = mdbcontainer1.jumpnetwork(0.45, 0.01, 0.01)
        crys_stars1 = DBStarSet(pdbcontainer1, mdbcontainer1, jset0, jset2, 1)

        DC_Si = crystal.Crystal(np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
                                , [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])

        pdbcontainer2 = pureDBContainer(DC_Si, 0, family)
        mdbcontainer2 = mixedDBContainer(DC_Si, 0, family)
        jset0 = pdbcontainer2.jumpnetwork(0.3, 0.01, 0.01)
        jset2 = mdbcontainer2.jumpnetwork(0.3, 0.01, 0.01)
        crys_stars2 = DBStarSet(pdbcontainer2, mdbcontainer2, jset0, jset2, 1)

        crys2d = crystal.Crystal(np.array([[1., 0.], [0., 1.5]]), [[np.array([0, 0]), np.array([0.5, 0.5])]], ["A"])

        o = np.array([0.1, 0.])
        famp02d = [o.copy()]
        family2d = [famp02d]
        pdbcontainer2d = pureDBContainer(crys2d, 0, family2d)
        mdbcontainer2d = mixedDBContainer(crys2d, 0, family2d)

        jset02d, jset22d = pdbcontainer2d.jumpnetwork(1.51, 0.01, 0.01), mdbcontainer2d.jumpnetwork(1.51, 0.01, 0.01)

        crys_stars3 = DBStarSet(pdbcontainer2d, mdbcontainer2d, jset02d, jset22d, Nshells=1)

        # Check that the stars are properly generated
        for crys_stars in [crys_stars1, crys_stars2, crys_stars3]:

            for star in crys_stars.stars[:crys_stars.mixedstartindex]:
                repr = star[0]
                considered_already = set([])
                count = 0
                for gdumb in crys_stars.pdbcontainer.G:
                    stnew = repr.gop(crys_stars.pdbcontainer, gdumb)[0]
                    stnew -= stnew.R_s
                    if stnew in star and not stnew in considered_already:
                        count += 1
                        considered_already.add(stnew)
                self.assertEqual(count, len(star))

            # test indexing
            for star, starind in zip(crys_stars.stars[:crys_stars.mixedstartindex],
                                     crys_stars.starindexed[:crys_stars.mixedstartindex]):
                self.assertEqual(len(star), len(starind))
                for state, stateind in zip(star, starind):
                    self.assertEqual(state, crys_stars.complexStates[stateind])

            for star, starind in zip(crys_stars.stars[crys_stars.mixedstartindex:],
                                     crys_stars.starindexed[crys_stars.mixedstartindex:]):
                for state, stateind in zip(star, starind):
                    self.assertEqual(state, crys_stars.mixedstates[stateind])

    def test_dicts(self):
        hcp_Mg = crystal.Crystal.HCP(0.3294, chemistry=["Mg"])
        fcc_Ni = crystal.Crystal.FCC(0.352, chemistry=["Ni"])
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]

        crys2d = crystal.Crystal(np.array([[1., 0.], [0., 1.5]]), [[np.array([0, 0]), np.array([0.5, 0.5])]], ["A"])

        o = np.array([0., 0.1])
        famp02d = [o.copy()]
        family2d = [famp02d]
        pdbcontainer2d = pureDBContainer(crys2d, 0, family2d)
        mdbcontainer2d = mixedDBContainer(crys2d, 0, family2d)

        jset02d, jset22d = pdbcontainer2d.jumpnetwork(1.51, 0.01, 0.01), mdbcontainer2d.jumpnetwork(1.51, 0.01, 0.01)

        crys_list = [hcp_Mg, fcc_Ni, DC_Si, crys2d]

        for struct, crys in enumerate(crys_list):

            if crys.dim == 2:
                crys_stars = DBStarSet(pdbcontainer2d, mdbcontainer2d, jset02d, jset22d, Nshells=1)
            else:
                pdbcontainer = pureDBContainer(crys, 0, family)
                mdbcontainer = mixedDBContainer(crys, 0, family)
                jset0 = pdbcontainer.jumpnetwork(0.45, 0.01, 0.01)
                jset2 = mdbcontainer.jumpnetwork(0.45, 0.01, 0.01)
                # 4.5 angst should cover atleast the nn distance in all the crystals
                # create starset
                crys_stars = DBStarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)

            # first, test the pure dictionary
            for key, value in crys_stars.complexIndexdict.items():
                self.assertEqual(key, crys_stars.complexStates[value[0]])
                self.assertTrue(crys_stars.complexStates[value[0]] in crys_stars.stars[value[1]])

            # Next, the mixed dictionary
            for key, value in crys_stars.mixedindexdict.items():
                self.assertEqual(key, crys_stars.mixedstates[value[0]])
                self.assertTrue(crys_stars.mixedstates[value[0]] in crys_stars.stars[value[1]])

            # Now test star2symlist
            for starind, star in enumerate(crys_stars.stars[:crys_stars.mixedstartindex]):
                symind = crys_stars.star2symlist[starind]
                for state in star:
                    db = state.db - state.db.R
                    # now get the symorlist index in which the dumbbell belongs
                    symind_other = crys_stars.pdbcontainer.invmap[db.iorind]
                    self.assertEqual(symind_other, symind, msg="\n{}".format(db))

            for starind, star in enumerate(crys_stars.stars[crys_stars.mixedstartindex:]):
                symind = crys_stars.star2symlist[starind]
                for state in star:
                    db = state.db - state.db.R
                    # now get the symorlist index in which the dumbbell belongs
                    symind_other = crys_stars.mdbcontainer.invmap[db.iorind]
                    self.assertEqual(symind_other, symind, msg="\n{}".format(db))

    def test_jumpnetworks(self):
        def inlist(jmp, jlist):
            return any(j == jmp for j in jlist)

        hcp_Mg = crystal.Crystal.HCP(0.3294, chemistry=["Mg"])
        fcc_Ni = crystal.Crystal.FCC(0.38, chemistry=["Ni"])
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]

        crys2d = crystal.Crystal(np.array([[1., 0.], [0., 1.5]]), [[np.array([0, 0]), np.array([0.5, 0.5])]], ["A"])


        crys_list = [DC_Si, hcp_Mg, fcc_Ni, crys2d]

        for struct, crys in enumerate(crys_list):
            print("Structure:{}\n  dimension:{}\n".format(struct, crys.dim))
            if crys.dim == 3:
                pdbcontainer = pureDBContainer(crys, 0, family)
                mdbcontainer = mixedDBContainer(crys, 0, family)
                jset0 = pdbcontainer.jumpnetwork(0.35, 0.01, 0.01)
                jset2 = mdbcontainer.jumpnetwork(0.35, 0.01, 0.01)
                # 4.5 angst should cover atleast the nn distance in all the crystals
                # create starset
                crys_stars = DBStarSet(pdbcontainer, mdbcontainer, jset0, jset2, Nshells=1)

            else:
                o = np.array([0., 0.1])
                famp02d = [o.copy()]
                family2d = [famp02d]
                pdbcontainer2d = pureDBContainer(crys, 0, family2d)
                mdbcontainer2d = mixedDBContainer(crys, 0, family2d)

                jset02d, jset22d = pdbcontainer2d.jumpnetwork(1.51, 0.01, 0.01), mdbcontainer2d.jumpnetwork(1.51, 0.01, 0.01)

                crys_stars = DBStarSet(pdbcontainer2d, mdbcontainer2d, jset02d, jset22d, Nshells=1)

            for jmplist in crys_stars.jnet0:
                for jmp in jmplist:
                    self.assertTrue(isinstance(jmp.state1, dumbbell), msg="\n{}".format(struct))
            ##TEST omega_1
            (omega1_network, omega1_indexed, omega1tag), om1types = crys_stars.jumpnetwork_omega1()
            for jlist, initdict in zip(omega1_indexed, omega1tag):
                for IS, jtag in initdict.items():
                    # go through the rows of the jtag:
                    for row in range(len(jtag)):
                        self.assertTrue(jtag[row][IS] == 1)
                        for column in range(len(crys_stars.complexStates) + len(crys_stars.mixedstates)):
                            if jtag[row][column] == -1:
                                self.assertTrue(any(i == IS and j == column for (i, j), dx in jlist))
                                # If any is true, then that means only one is true, since a jump b/w two states is
                                # present only once.

            rotset = set([])  # Here we will store the rotational jumps in the network
            rotInd = []
            for x in range(len(omega1_network)):
                # select any jump from this list at random. Idea is that we must get back the same jump list.
                y = np.random.randint(0, len(omega1_network[x]))
                jmp = omega1_network[x][y]
                # First, check that the solute does not move and is at the origin
                self.assertTrue(jmp.state1.i_s == jmp.state2.i_s)
                self.assertTrue(np.array_equal(jmp.state1.R_s, np.zeros(crys_stars.crys.dim, dtype=int)))
                self.assertTrue(np.array_equal(jmp.state2.R_s, np.zeros(crys_stars.crys.dim, dtype=int)))

                # Next, collect rotational jumps for checking later
                if np.allclose(DB_disp(crys_stars.pdbcontainer, jmp.state1, jmp.state2), 0.,
                                atol=crys_stars.pdbcontainer.crys.threshold):
                    for rotjmp in omega1_network[x]:
                        rotset.add(rotjmp)
                    rotInd.append(x)
                    continue
                # we'll test the redundance of rotation jumps separately.
                jlist = []
                # reconstruct the list using the selected jump, without using hash tables (sets)
                for gdumb in crys_stars.pdbcontainer.G:
                    # shift the states back to the origin unit cell
                    state1new, flip1 = jmp.state1.gop(crys_stars.pdbcontainer, gdumb)
                    state2new, flip2 = jmp.state2.gop(crys_stars.pdbcontainer, gdumb)
                    jnew = jump(state1new - state1new.R_s, state2new - state2new.R_s, jmp.c1 * flip1, jmp.c2 * flip2)
                    if not any(jnew == j for j in jlist):
                        jlist.append(jnew)
                        jlist.append(-jnew)
                self.assertEqual(len(jlist), len(omega1_network[x]))

            # Now check redundant rotation removal
            for rotjmp in rotset:
                j_equiv = jump(rotjmp.state1, rotjmp.state2, -rotjmp.c1, -rotjmp.c2)
                self.assertFalse(j_equiv in rotset)

            # Now check symmetries of rotation jumps
            for x in rotInd:
                # select any jump from this list at random. Idea is that we must get back the same jump list.
                y = np.random.randint(0, len(omega1_network[x]))
                jmp = omega1_network[x][y]
                # First, check that the solute does not move and is at the origin
                self.assertTrue(jmp.state1.i_s == jmp.state2.i_s)
                self.assertTrue(np.array_equal(jmp.state1.R_s, np.zeros(crys_stars.crys.dim, dtype=int)))
                self.assertTrue(np.array_equal(jmp.state2.R_s, np.zeros(crys_stars.crys.dim, dtype=int)))
                jlist = []
                for gdumb in crys_stars.pdbcontainer.G:
                    # shift the states back to the origin unit cell
                    state1new, flip1 = jmp.state1.gop(crys_stars.pdbcontainer, gdumb)
                    state2new, flip2 = jmp.state2.gop(crys_stars.pdbcontainer, gdumb)
                    jnew = jump(state1new - state1new.R_s, state2new - state2new.R_s, jmp.c1 * flip1, jmp.c2 * flip2)
                    if not any(jnew == j for j in jlist) and jnew in rotset:
                        jlist.append(jnew)
                        jlist.append(-jnew)
                self.assertEqual(len(jlist), len(omega1_network[x]))


            # See that irrespective of solute location, if the jumps of the dumbbells are the same, then the jump type
            # is also the same
            for i, jlist1 in enumerate(omega1_network):
                for j, jlist2 in enumerate(omega1_network[:i]):
                    for j1 in jlist1:
                        for j2 in jlist2:
                            if j1.state1.db == j2.state1.db and j1.state2.db == j2.state2.db:
                                if j1.c1 == j2.c1 and j1.c2 == j2.c2:
                                    self.assertTrue(om1types[i] == om1types[j],
                                                    msg="{},{}\n{}\n{}".format(i, j, j1, j2))

            if crys.dim == 3:
                cut43 = 0.35
            else:
                cut43 = 1.51

            omega43, omega4, omega3 = crys_stars.jumpnetwork_omega34(cut43, 0.01, 0.01, 0.01)
            omega43_all, omega4_network, omega3_network = omega43[0], omega4[0], omega3[0]
            omega43_all_indexed, omega4_network_indexed, omega3_network_indexed = omega43[1], omega4[1], omega3[1]
            omega4tag, omega3tag = omega4[2], omega3[2]
            self.assertEqual(len(omega4_network), len(omega3_network))
            for jl4, jl3 in zip(omega4_network, omega3_network):
                self.assertEqual(len(jl3), len(jl4))
                for j3, j4 in zip(jl3, jl4):
                    self.assertEqual(j3, -j4)
                    self.assertEqual(j3.c1, -1)
                    self.assertEqual(j4.c2, -1)

            for jlist_all, j4list in zip(omega43_all, omega4_network):
                for jmp1, jmp2 in zip(jlist_all[::2], j4list):
                    self.assertEqual(jmp1, jmp2)

            for jlist_all, j3list in zip(omega43_all, omega3_network):
                for jmp1, jmp2 in zip(jlist_all[1::2], j3list):
                    self.assertEqual(jmp1, jmp2)

            ##TEST omega3 and omega4
            # test that the tag lists have proper length
            self.assertEqual(len(omega4tag), len(omega4_network))
            self.assertEqual(len(omega3tag), len(omega3_network))
            omeg34list = [omega3_network, omega4_network]
            for i, omegalist in enumerate(omeg34list):
                for x in range(len(omegalist)):
                    y = np.random.randint(0, len(omegalist[x]))
                    jmp = omegalist[x][y]
                    jlist = []
                    if i == 0:  # then build omega3
                        self.assertTrue(jmp.state1.is_zero(crys_stars.mdbcontainer))
                        for gdumb, gcrys in crys_stars.pdbcontainer.G_crys.items():
                            for gd, g in crys_stars.mdbcontainer.G_crys.items():
                                if g == gcrys:
                                    mgdumb = gd
                            state1new = jmp.state1.gop(crys_stars.mdbcontainer, mgdumb, complex=False)
                            state2new, flip2 = jmp.state2.gop(crys_stars.pdbcontainer, gdumb)
                            jnew = jump(state1new - state1new.R_s, state2new - state2new.R_s, -1, jmp.c2 * flip2)
                            if not inlist(jnew, jlist):
                                jlist.append(jnew)
                    else:  # build omega4
                        self.assertTrue(jmp.state2.is_zero(crys_stars.mdbcontainer))
                        for gdumb, gcrys in crys_stars.pdbcontainer.G_crys.items():
                            for gd, g in crys_stars.mdbcontainer.G_crys.items():
                                if g == gcrys:
                                    mgdumb = gd
                            state1new, flip1 = jmp.state1.gop(crys_stars.pdbcontainer, gdumb)
                            state2new = jmp.state2.gop(crys_stars.mdbcontainer, mgdumb, complex=False)
                            jnew = jump(state1new - state1new.R_s, state2new - state2new.R_s, jmp.c1 * flip1, -1)
                            if not inlist(jnew, jlist):
                                jlist.append(jnew)
                    self.assertEqual(len(jlist), len(omegalist[x]), msg="{}".format(omegalist[x][0]))

            ##Test indexing of the jump networks
            # First, omega_1
            for jlist, jindlist in zip(omega1_network, omega1_indexed):
                for jmp, indjmp in zip(jlist, jindlist):
                    self.assertTrue(jmp.state1 == crys_stars.complexStates[indjmp[0][0]])
                    self.assertTrue(jmp.state2 == crys_stars.complexStates[indjmp[0][1]])
            # Next, omega34
            for jlist, jindlist in zip(omega4_network, omega4_network_indexed):
                for jmp, indjmp in zip(jlist, jindlist):
                    self.assertTrue(jmp.state1 == crys_stars.complexStates[indjmp[0][0]])
                    self.assertTrue(jmp.state2 == crys_stars.mixedstates[indjmp[0][1]])

            for jlist, jindlist in zip(omega3_network, omega3_network_indexed):
                for jmp, indjmp in zip(jlist, jindlist):
                    # print(jmp.state1)
                    # print()
                    # print(crys_stars.mixedstates[indjmp[0][0]])
                    self.assertTrue(jmp.state1 == crys_stars.mixedstates[indjmp[0][0]], msg="{}".format(struct))
                    self.assertTrue(jmp.state2 == crys_stars.complexStates[indjmp[0][1]])
            # testing the tags
            # First, omega4
            for jlist, initdict in zip(omega4_network_indexed, omega4tag):
                for IS, jtag in initdict.items():
                    # go through the rows of the jtag:
                    for row in range(len(jtag)):
                        self.assertTrue(jtag[row][IS] == 1)
                        for column in range(len(crys_stars.complexStates) + len(crys_stars.mixedstates)):
                            if jtag[row][column] == -1:
                                self.assertTrue(
                                    any(i == IS and j == column - len(crys_stars.complexStates) for (i, j), dx in
                                        jlist))
            # Next, omega3
            for jlist, initdict in zip(omega3_network_indexed, omega3tag):
                for IS, jtag in initdict.items():
                    # go through the rows of the jtag:
                    for row in range(len(jtag)):
                        self.assertTrue(jtag[row][IS + len(crys_stars.complexStates)] == 1)
                        for column in range(len(crys_stars.complexStates) + len(crys_stars.mixedstates)):
                            if jtag[row][column] == -1:
                                self.assertTrue(any(i == IS and j == column for (i, j), dx in jlist))
            # Next, omega2 to mixedstates
            jnet2, jnet2stateindex = crys_stars.jnet2, crys_stars.jnet2_ind
            for i in range(len(jnet2)):
                self.assertEqual(len(jnet2[i]), len(jnet2stateindex[i]))
                for jpair, jind in zip(jnet2[i], jnet2stateindex[i]):
                    IS = crys_stars.mixedstates[jind[0][0]]
                    FS = crys_stars.mixedstates[jind[0][1]]
                    self.assertEqual(IS, jpair.state1 - jpair.state1.R_s,
                                     msg="\n{} not equal to {}".format(IS, jpair.state1))
                    self.assertEqual(FS, jpair.state2 - jpair.state2.R_s,
                                     msg="\n{} not equal to {}".format(FS, jpair.state2))

            for jlist, initdict in zip(jnet2stateindex, crys_stars.jtags2):
                for IS, jtag in initdict.items():
                    # go through the rows of the jtag:
                    for row in range(len(jtag)):
                        # The column corresponding to the intial state must have 1.
                        self.assertTrue(jtag[row][IS + len(crys_stars.complexStates)] == 1 or jtag[row][
                            IS + len(crys_stars.complexStates)] == 0,
                                        msg="{}".format(jtag[row][IS + len(crys_stars.complexStates)]))
                        # the zero appears when the intial and final states are the same (but just translated in the lattice) so that they have the same periodic eta vector
                        for column in range(len(crys_stars.complexStates) + len(crys_stars.mixedstates)):
                            if jtag[row][column] == -1:
                                self.assertTrue(
                                    any(i == IS and j == column - len(crys_stars.complexStates) for (i, j), dx in
                                        jlist))

    def test_om1types(self):
        """
        This is an expensive test, so did not include in previous section
        """
        hcp_Mg=crystal.Crystal.HCP(0.3294,chemistry=["Mg"])
        fcc_Ni = crystal.Crystal.FCC(0.352,chemistry=["Ni"])
        latt = np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])*0.55
        DC_Si = crystal.Crystal(np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55,
                                [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        crys_list = [DC_Si, hcp_Mg, fcc_Ni]
        for crys in crys_list:
            pdbcontainer = pureDBContainer(crys, 0, family)
            mdbcontainer = mixedDBContainer(crys, 0, family)
            jset0 = pdbcontainer.jumpnetwork(0.24, 0.01, 0.01)
            jset2 = mdbcontainer.jumpnetwork(0.24, 0.01, 0.01)
            # 4.5 angst should cover atleast the nn distance in all the crystals
            # create starset
            crys_stars = DBStarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)
            (omega1_network, omega1_indexed, omega1tag), om1types = crys_stars.jumpnetwork_omega1()
            for i, jlist1 in enumerate(omega1_network):
                for j, jlist2 in enumerate(omega1_network):
                    if i == j: continue
                    for j1 in jlist1:
                        for j2 in jlist2:
                            if j1.state1.db == j2.state1.db and j1.state2.db == j2.state2.db:
                                if j1.c1 == j2.c1 and j1.c2 == j2.c2:
                                    self.assertTrue(om1types[i] == om1types[j],
                                                    msg="{},{}\n{}\n{}".format(i, j, j1, j2))

    def test_sort_stars(self):
        DC_Si = crystal.Crystal(np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55,
                                [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        pdbcontainer = pureDBContainer(DC_Si, 0, family)
        mdbcontainer = mixedDBContainer(DC_Si, 0, family)
        jset0 = pdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        jset2 = mdbcontainer.jumpnetwork(0.3, 0.01, 0.01)
        # 4.5 angst should cover atleast the nn distance in all the crystals
        # create starset
        crys_stars = DBStarSet(pdbcontainer, mdbcontainer, jset0, jset2, 1)
        dx_list = []
        for sts in zip(crys_stars.stars[:crys_stars.mixedstartindex]):
            st0 = sts[0][0]
            sol_pos = crys_stars.crys.unit2cart(st0.R_s, crys_stars.crys.basis[crys_stars.chem][st0.i_s])
            db_pos = crys_stars.crys.unit2cart(st0.db.R, crys_stars.crys.basis[crys_stars.chem][crys_stars.pdbcontainer.iorlist[st0.db.iorind][0]])
            dx = np.linalg.norm(db_pos - sol_pos)
            dx_list.append(dx)
        self.assertTrue(np.allclose(np.array(dx_list), np.array(sorted(dx_list))),
                        msg="\n{}\n{}".format(dx_list, sorted(dx_list)))
