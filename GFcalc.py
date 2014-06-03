import numpy as np

def DFTfunc(NNvect, rates):
   """
   Returns a Fourier-transform function given the NNvect and rates
   
   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   return lambda q: np.sum(np.cos(np.dot(NNvect,q))*rates)-np.sum(rates)

def D2(NNvect, rates):
   """
   Construct the diffusivity matrix (small q limit of Fourier transform
   as a second derivative).
   Returns a 3x3 matrix that can be dotted into q to get FT.

   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   # return np.zeros((3,3))
   return 0.5*np.dot(NNvect.T*rates, NNvect)
   
def D4(NNvect, rates):
   """
   Construct the discontinuity matrix (fourth derivative wit respect to q of
   Fourier transform).
   Returns a 3x3x3x3 matrix that can be dotted into q to get the FT.

   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   D4 = np.zeros((3,3,3,3))
   for a in xrange(3):
      for b in xrange(3):
         for c in xrange(3):
            for d in xrange(3):
               D4[a,b,c,d] = 1./24. * sum(NNvect[:,a]*NNvect[:,b]*NNvect[:,c]*NNvect[:,d]*rates[:])
   return D4

def calcDE(D2):
   """
   Takes in the D2 matrix (assumed to be real, symmetric) and diagonalizes it
   returning the eigenvalues (d_i) and corresponding normalized eigenvectors (e_i).
   Returns di[3], ei[3,3], where ei[i,:] is the eigenvector for di[i]. NOTE: this is
   the transposed version of what eigh returns.
   
   Parameters
   ----------
   D2[3,3]: symmetric, real matrix from D2()
   """

   di, ei=np.linalg.eigh(D2)
   return di, ei.T

def invertD2(D2):
   """
   Takes in the matrix D2, returns its inverse (which gets used repeatedly).

   Parameters
   ----------
   D2[3,3]: symmetric, real matrix from D2()
   """

   return np.linalg.inv(D2)

def unorm(di, ei, x):
   """
   Takes the eigenvalues di, eigenvectors ei, and the vector x, and returns the
   normalized u vector, along with its magnitude. These are the key elements needed
   in *all* of the Fourier transform expressions to follow.

   Returns: ui[3], umagn

   Parameters
   ----------
   di[3]:   eigenvalues of D2
   ei[3,3]: eigenvectors of D2 (ei[i,:] == ith eigenvector)
   x[3]:    cartesian position vector
   """

   ui = np.zeros(3)
   umagn = 0
   if (np.dot(x,x)>0):
      ui = np.dot(ei, x)/np.sqrt(di)
      umagn = np.sqrt(np.dot(ui,ui))
      ui /= umagn
   return ui, umagn

