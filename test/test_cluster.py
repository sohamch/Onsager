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


if __name__ == '__main__':
    unittest.main()