import numpy as np
from collections import namedtuple


# Single dumbbell state representer class.
# 1. Format - 'i o R c' -> basis index, orientation, lattice vector, active atom indicator
# 2. Should be able to check if two dumbbell states are identical
# 3. Should be able to add a jump to a dumbbell state.
# 4. Should be able to apply a given group operation (crystal specified) to a dumbbell.

class dumbbell(namedtuple('dumbbell', 'iorind R')):

    def __eq__(self, other):
        # zero=np.zeros(len(self.o))
        true_class = isinstance(other, self.__class__)
        c1 = true_class and (self.iorind == other.iorind and np.allclose(self.R, other.R, atol=1e-8))
        return c1

    def __ne__(self, other):
        return not self.__eq__(other)

    # def flip(self, container):
    #     # Check crystal module for how dumbbells are rotated.
    #     return self.__class__(container.fliplist[self.iorind], self.R)

    def __hash__(self):
        # o = np.round(self.o,6)
        # return hash((self.i,o[0],o[1]*5,o[2],self.R[0],self.R[1],self.R[2]))
        return hash((self.iorind,) + tuple(self.R))

    def gop(self, container, gdumb, pure=True):

        # If we have a pure dumbbell, we return the result of the groupop, as well as the flip indicator
        # Otherwise, just return the new dumbbell
        i, o = container.iorlist[self.iorind]
        # Dealing with a pure dumbbell
        R_new, (ch, i_new) = container.crys.g_pos(container.G_crys[gdumb], self.R, (container.chem, i))
        if not i_new == container.iorlist[gdumb.indexmap[0][self.iorind]][0]:
            raise ValueError("Gdumb and G not consistent")
        newind = gdumb.indexmap[0][self.iorind]
        if pure:
            flipind = container.gflip(gdumb, self.iorind)
            return self.__class__(newind, R_new), flipind
        else:
            return self.__class__(newind, R_new)

    def __add__(self, other):
        if not isinstance(other, np.ndarray):
            raise TypeError("Can only add a lattice translation to a dumbbell")
        if not len(other) == len(self.R):
            raise TypeError("Can add only a lattice translation (vector) to a dumbbell")
        for x in other:
            if not isinstance(x, np.dtype(int).type):
                raise TypeError("Can add only a lattice translation vector (integer components) to a dumbbell")

        return self.__class__(self.iorind, self.R + other)

    def __sub__(self, other):
        return self.__add__(-other)


# A Pair obect (that represents a dumbbell-solute state) should have the following attributes:
# 1. It should have the locations of the solute and dumbbell.
# 2. A pair dbect contains information regarding which atom in the dumbbell is going to jump.
# 3. We should be able to apply Group operations to it to generate new pairs.
# 4. We should be able to add a Jump dbect to a Pair dbect to create another pair dbect.
# 5. We should be able to test for equality.
# 6. Applying group operations should be able to return the correct results for seperated and mixed dumbbell pairs.

class SdPair(namedtuple('SdPair', "i_s R_s db")):
    def __eq__(self, other):
        true_class = isinstance(other, self.__class__)
        true_solute = self.i_s == other.i_s and np.allclose(self.R_s, other.R_s, atol=1e-8)
        true_db = self.db == other.db
        return true_class and true_solute and true_db

    def __ne__(self, other):
        return not self.__eq__(other)

    # def flip(self, container):
    #     # negation is used to flip the orientation vector
    #     return self.__class__(self.i_s, self.R_s, self.db.flip(container))

    def __hash__(self):
        return hash((self.i_s, self.R_s[0], hash(self.db)))
        # return id(self)

    def gop(self, container, gdumb, complex=True):  # apply group operation
        # If we have a complex, return a flip indicator as well, else, just return the new pair
        R_s_new, (ch, i_s_new) = container.crys.g_pos(container.G_crys[gdumb], self.R_s,
                                                      (container.chem, self.i_s))
        if complex:
            dbnew, flip = self.db.gop(container, gdumb, pure=True)
            return self.__class__(i_s_new, R_s_new, dbnew), flip
        else:
            dbnew = self.db.gop(container, gdumb, pure=False)
            return self.__class__(i_s_new, R_s_new, dbnew)

    def is_zero(self, container):
        """
        To check if solute and dumbbell are at the same site
        """
        return self.i_s == container.iorlist[self.db.iorind][0] and np.allclose(self.R_s, self.db.R, atol=1e-8)

    def addjump(self, j, mixed=False):

        if not mixed:
            if isinstance(j.state1, self.__class__):
                raise TypeError("Only dumbbell -> dumbbell transitions can be added to complexes")
            if not self.db.iorind == j.state1.iorind:
                raise ArithmeticError("Incompatible starting dumbbell configurations")
            if not np.allclose(j.state1.R, 0):
                raise ValueError("Initial dumbbell not at origin unit cell")
            db2 = dumbbell(j.state2.iorind, self.db.R + j.state2.R - j.state1.R)
            return self.__class__(self.i_s, self.R_s, db2)

        if mixed:
            if not isinstance(j.state1, SdPair):
                raise TypeError("Only pair -> pair transitions can be added to mixed dumbbells")
            if not self.db.iorind == j.state1.db.iorind:
                raise ArithmeticError("Incompatible starting dumbbell configurations")
            db2 = dumbbell(j.state2.db.iorind, self.db.R + j.state2.db.R - j.state1.db.R)
            return SdPair(j.state2.i_s, db2.R, db2)

    def __add__(self, other):

        """
        Adding a translation to a solute-dumbbell pair shifts both the solute and the dumbbell
        by the same translation vector.
        Usually this will be used to bring the solute back to the origin unit cell and shift the dumbbell accordingly.
        """
        if not isinstance(other, np.ndarray):
            raise TypeError("Can only add a lattice translation to a dumbbell")

        if not len(other) == len(self.R_s):
            raise TypeError("Can add only a lattice translation (vector) to a dumbbell")

        for x in other:
            if not isinstance(x, np.dtype(int).type):
                raise TypeError("Can add only a lattice translation vector (integer components) to a dumbbell")

        return self.__class__(self.i_s, self.R_s + other, self.db + other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __xor__(self, other):
        """
        Creates a connector object, from dumbbell state of self to dumbbell state of other.
        """
        if not type(self) == type(other):
            raise ValueError("Can only xor between two SdPair objects")

        if self.i_s != other.i_s or not np.allclose(self.R_s, other.R_s):
            raise ArithmeticError("can only connect states with same solute location.")

        return connector(self.db - self.db.R, other.db - self.db.R)


# Jump obects are rather simple, contain just initial and final orientations
# dumbell/pair objects are not aware of jump objects.

class jump(namedtuple('jump', 'state1 state2 c1 c2')):

    def __init__(self, state1, state2, c1, c2):
        # Do Type checking of input stateects
        if not isinstance(self.state2, self.state1.__class__):
            raise TypeError("Incompatible Initial and final states. They must be of the same type.")

    def __eq__(self, other):
        return (self.state1 == other.state1 and self.state2 == other.state2 and
                self.c1 == other.c1 and self.c2 == other.c2)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # return hash((self.state1,self.state2,self.c1,self.c2))
        return hash((hash(self.state1), hash(self.state2), self.c1, self.c2))
        # return id(self)

    def __add__(self, other):
        # Do type checking of input operands and jump states
        if not isinstance(other, dumbbell):
            raise TypeError("For now jumps can only be added to dumbbell objects.")
        if not self.state1.iorind == other.iorind:
            raise ArithmeticError("Operand dumbbell and initial dumbbell of jump must have same configuration.")
        return dumbbell(self.state2.iorind, other.R + self.state2.R - self.state1.R)  # Add the lattice transition in.

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        # negation is used to flip the transition in the opposite direction
        return self.__class__(self.state2, self.state1, self.c2, self.c1)

    def __str__(self):
        strrep = None
        if isinstance(self.state1, SdPair):
            strrep = "Jump object:\nInitial state:\n\t"
            strrep += "Solute loctation:basis index = {}, lattice vector = {}\n\t".format(self.state1.i_s,
                                                                                          self.state1.R_s)

            strrep += "dumbbell : (i, or) index = {}, lattice vector = {}\n".format(self.state1.db.iorind,
                                                                                    self.state1.db.R)
            strrep += "Final state:\n\t"
            strrep += "Solute loctation :basis index = {}, lattice vector = {}\n\t".format(self.state2.i_s,
                                                                                           self.state2.R_s)

            strrep += "dumbbell : (i, or) index = {}, lattice vector = {}\n".format(self.state2.db.iorind,
                                                                                    self.state2.db.R)
            strrep += "Jumping from c1 = {} to c2 = {}".format(self.c1, self.c2)

        if isinstance(self.state1, dumbbell):
            strrep = "Jump object:\nInitial state:\n\t"
            strrep += "dumbbell : (i, or) index = {}, lattice vector = {}\n".format(self.state1.iorind,
                                                                                    self.state1.R)
            strrep += "Final state:\n\t"
            strrep += "dumbbell : (i, or) index = {}, lattice vector = {}".format(self.state2.iorind,
                                                                                    self.state2.R)
            strrep += "\nJumping from c1 = {} to c2 = {}\n".format(self.c1, self.c2)

        return strrep


class connector(namedtuple('connector', 'state1 state2')):
    """
    An object that simple connects two states.
    Similar to the jump object, but does not contain information regarding connecting path.
    Checks compatibility of connections as well.
    """

    def __init__(self, state1, state2):
        # Check compatibility
        if not (isinstance(self.state1, dumbbell) and isinstance(self.state2, dumbbell)):
            raise TypeError("Incompatible Initial and final states. They must be of the dumbbell type.")

        # Check correctness
        if not np.allclose(state1.R, 0.):
            raise ValueError("The initial dumbbell in a connector must always be at the origin unit cell")

    def __eq__(self, other):
        return self.state1 == other.state1 and self.state2 == other.state2

    def __hash__(self):
        return hash((self.state1, self.state2))

    def __neg__(self):
        return self.__class__(self.state2 - self.state2.R, self.state1 - self.state2.R)

    def gop(self, container, gdumb, pure=True):

        state1new = self.state1.gop(container, gdumb, pure=pure)
        state2new = self.state2.gop(container, gdumb, pure=pure)
        if pure:
            db2new = state2new[0] - state1new[0].R
            db1new = state1new[0] - state1new[0].R
            return self.__class__(db1new, db2new)
        else:
            return self.__class__(state1new - state1new.R, state2new - state1new.R)

    def shift(self):
        """
        :return: returns a translated connector with the initial state at the origin unit cell.
        """
        return self.__class__(self.state1 - self.state1.R, self.state2 - self.state1.R)
