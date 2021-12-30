import numpy as np
import pytest

from latte.functional.disentanglement.mutual_info import dmig, mig
from latte.metrics.core.disentanglement import *


class TestMIG:
    def test_mig_noreg_continuous(self):
        mod = MutualInformationGap()

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randn(16, 3)

            zl.append(z)
            al.append(a)

            mod.update_state(z, a)

        val = mod.compute()

        np.testing.assert_allclose(
            val, mig(np.concatenate(zl, axis=0), np.concatenate(al, axis=0))
        )

    def test_mig_reg_continuous(self):
        mod = MutualInformationGap(reg_dim=[2, 3, 4])

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randn(16, 3)

            zl.append(z)
            al.append(a)

            mod.update_state(z, a)

        val = mod.compute()

        np.testing.assert_allclose(
            val,
            mig(
                np.concatenate(zl, axis=0),
                np.concatenate(al, axis=0),
                reg_dim=[2, 3, 4],
            ),
        )

    def test_mig_noreg_discrete(self):
        mod = MutualInformationGap(discrete=True)

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randint(16, size=(16, 3))

            zl.append(z)
            al.append(a)

            mod.update_state(z, a)

        val = mod.compute()

        np.testing.assert_allclose(
            val,
            mig(np.concatenate(zl, axis=0), np.concatenate(al, axis=0), discrete=True),
        )

    def test_mig_reg_discrete(self):
        mod = MutualInformationGap(reg_dim=[2, 3, 4], discrete=True)

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randint(16, size=(16, 3))

            zl.append(z)
            al.append(a)

            mod.update_state(z, a)

        val = mod.compute()

        np.testing.assert_allclose(
            val,
            mig(
                np.concatenate(zl, axis=0),
                np.concatenate(al, axis=0),
                reg_dim=[2, 3, 4],
                discrete=True,
            ),
        )


class TestDependencyAware:
    def test_dmig(self):
        mod = DependencyAwareMutualInformationGap()

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randn(16, 3)

            zl.append(z)
            al.append(a)

            mod.update_state(z, a)

        val = mod.compute()

        np.testing.assert_allclose(
            val, dmig(np.concatenate(zl, axis=0), np.concatenate(al, axis=0))
        )

    def test_dlig(self):
        mod = DependencyAwareLatentInformationGap()

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randn(16, 3)

            zl.append(z)
            al.append(a)

            mod.update_state(z, a)

        val = mod.compute()

        np.testing.assert_allclose(
            val, dlig(np.concatenate(zl, axis=0), np.concatenate(al, axis=0))
        )

    def test_xmig(self):
        mod = DependencyBlindMutualInformationGap()

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randn(16, 3)

            zl.append(z)
            al.append(a)

            mod.update_state(z, a)

        val = mod.compute()

        np.testing.assert_allclose(
            val, xmig(np.concatenate(zl, axis=0), np.concatenate(al, axis=0))
        )


def test_aliases():
    assert MutualInformationGap == MIG
    assert DependencyAwareMutualInformationGap == DMIG
    assert DependencyAwareLatentInformationGap == DLIG
    assert DependencyBlindMutualInformationGap == XMIG
