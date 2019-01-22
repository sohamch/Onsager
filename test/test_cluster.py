"""
Unit tests for star, double-star and vector-star generation and indexing,
rebuilt to use crystal
"""

__author__ = 'Dallas R. Trinkle'

#

import unittest
import onsager.crystal as crystal
import numpy as np
import onsager.cluster as cluster


class ClusterSiteTests(unittest.TestCase):
    """Tests of the ClusterSite class"""
    longMessage = False

    def testClusterSiteType(self):
        """Can we make cluster sites?"""
        site = cluster.ClusterSite((0,0), np.array([0,0,0]))
        self.assertIsInstance(site, cluster.ClusterSite)

    def testNegation(self):
        """Can we negate (and equate) cluster sites?"""
        s1 = cluster.ClusterSite((0,0), np.array([1,0,0]))
        s2 = cluster.ClusterSite((0,0), np.array([-1,0,0]))
        self.assertNotEqual(s1, s2)
        self.assertEqual(s1, -s2)

    def testAddition(self):
        """Can we add lattice vectors to a site?"""
        s1 = cluster.ClusterSite((0,0), np.array([0,0,0]))
        s2 = cluster.ClusterSite((0,0), np.array([1,0,0]))
        s3 = cluster.ClusterSite((0,0), np.array([-1,0,0]))
        v1 = np.array([1,0,0])
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertEqual(s1+v1, s2)
        self.assertNotEqual(s1-v1, s2)
        self.assertEqual(s1-v1, s3)
        self.assertNotEqual(s1+v1, s3)

class ClusterTests(unittest.TestCase):
    """Tests of the Cluster class"""
    longMessage = False

    def testMakeCluster(self):
        """Can we make a cluster?"""
        s = cluster.ClusterSite((0,0), np.array([0,0,0]))
        cl = cluster.Cluster([s])
        self.assertIsInstance(cl, cluster.Cluster)
        Rlist = [np.array([0,0,0]), np.array([1,0,0]), np.array([-1,0,0])]
        cl = cluster.Cluster(cluster.ClusterSite((0, n), R)
                             for n, R in enumerate(Rlist))
        self.assertIsInstance(cl, cluster.Cluster)

    def testEquality(self):
        """Equality tests"""
        s1 = cluster.ClusterSite((0,0), np.array([0,0,0]))
        s2 = cluster.ClusterSite((0,0), np.array([1,0,0]))
        c1 = cluster.Cluster([s1])
        c2 = cluster.Cluster([s2])
        c3 = cluster.Cluster([s1, s2])
        c4 = cluster.Cluster([s2, s1])
        self.assertEqual(c1, c2)
        self.assertNotEqual(c1, c3)
        self.assertEqual(c3, c4)

    def testAddition(self):
        """Making a cluster via addition"""
        s1 = cluster.ClusterSite((0,0), np.array([0,0,0]))
        s2 = cluster.ClusterSite((0,0), np.array([1,0,0]))
        c1 = cluster.Cluster([s1])
        c2 = c1 + s2
        c3 = cluster.Cluster([s2, s1])
        self.assertEqual(c2, c3)

    def testHash(self):
        """Can we make a set of clusters?"""
        s1 = cluster.ClusterSite((0,0), np.array([0,0,0]))
        s2 = cluster.ClusterSite((0,0), np.array([1,0,0]))
        c1 = cluster.Cluster([s1])
        c2 = cluster.Cluster([s2])
        c3 = c1 + s2
        set1 = set([c1])
        set2 = set([c1, c2])
        set3 = set([c1, c3])
        set4 = set([c2, c3])
        self.assertEqual(set1, set2)
        self.assertNotEqual(set1, set3)
        self.assertEqual(set3, set4)

    def testHCPGroupOp(self):
        """Testing group operations on our clusters (HCP)"""
        HCP = crystal.Crystal.HCP(1., chemistry='HCP')
        s1 = cluster.ClusterSite((0, 0), np.array([0, 0, 0]))
        s2 = cluster.ClusterSite((0, 1), np.array([0, 0, 0]))
        s3 = cluster.ClusterSite((0, 0), np.array([1, 0, 0]))

        cl = cluster.Cluster([s1])
        clusterset = set([cl.g(HCP, g) for g in HCP.G])
        self.assertEqual(len(clusterset), 2)

        cl = cluster.Cluster([s1, s2])
        clusterset = set([cl.g(HCP, g) for g in HCP.G])
        self.assertEqual(len(clusterset), 6)

        cl = cluster.Cluster([s1, s3])
        clusterset = set([cl.g(HCP, g) for g in HCP.G])
        self.assertEqual(len(clusterset), 6)

    def testFCCGroupOp(self):
        """Testing group operations on our clusters (FCC)"""
        FCC = crystal.Crystal.FCC(1., chemistry='FCC')
        s1 = cluster.ClusterSite.fromcryscart(FCC, np.array([0, 0, 0]))
        s2 = cluster.ClusterSite.fromcryscart(FCC, np.array([0., 0.5, 0.5]))
        s3 = cluster.ClusterSite.fromcryscart(FCC, np.array([0.5, 0., 0.5]))
        s4 = cluster.ClusterSite.fromcryscart(FCC, np.array([0.5, 0.5, 0.]))
        s5 = cluster.ClusterSite.fromcryscart(FCC, np.array([0., 0.5, -0.5]))

        # only one way to make a single site:
        cl = cluster.Cluster([s1])
        clusterset = set([cl.g(FCC, g) for g in FCC.G])
        self.assertEqual(1, len(clusterset), msg='Failure on single site')

        # six ways to make a NN pair: multiplicity of <110>, divided by 2
        cl = cluster.Cluster([s1, s2])
        clusterset = set([cl.g(FCC, g) for g in FCC.G])
        self.assertEqual(6, len(clusterset), msg='Failure on NN pair')

        # eight ways to make a NN triplet: multiplicity of <111> (the plane normal of the face)
        cl = cluster.Cluster([s1, s2, s3])
        clusterset = set([cl.g(FCC, g) for g in FCC.G])
        self.assertEqual(8, len(clusterset), msg='Failure on NN triplet')

        # twelve ways to make a "wide" triplet: multiplicity of <100> times two
        cl = cluster.Cluster([s1, s2, s5])
        clusterset = set([cl.g(FCC, g) for g in FCC.G])
        self.assertEqual(12, len(clusterset), msg='Failure on wide NN triplet')

        # two ways to make our tetrahedron
        cl = cluster.Cluster([s1, s2, s3, s4])
        clusterset = set([cl.g(FCC, g) for g in FCC.G])
        self.assertEqual(2, len(clusterset), msg='Failure on NN quad')

    def testmakeclustersFCC(self):
        """Does makeclusters perform as expected? FCC"""
        FCC = crystal.Crystal.FCC(1., chemistry='FCC')
        clusterexp = cluster.makeclusters(FCC, 0.8, 4)
        self.assertEqual(4, len(clusterexp))
        self.assertEqual([1, 6, 8, 2], [len(clset) for clset in clusterexp])



if __name__ == '__main__':
    unittest.main()
