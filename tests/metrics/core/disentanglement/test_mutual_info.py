import pytest
from latte.metrics.core.disentanglement.mutual_info import MutualInformationGap
import numpy as np
from latte.functional.disentanglement.mutual_info import mig


class TestMIG:
    def test_mig_noreg_continuous(self):
        mig_mod = MutualInformationGap()

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randn(16, 3)

            zl.append(z)
            al.append(a)

            mig_mod.update_state(z, a)

        mig_val = mig_mod.compute()

        np.testing.assert_allclose(
            mig_val, mig(np.concatenate(zl, axis=0), np.concatenate(al, axis=0))
        )

    def test_mig_reg_continuous(self):
        mig_mod = MutualInformationGap(reg_dim=[2, 3, 4])

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randn(16, 3)

            zl.append(z)
            al.append(a)

            mig_mod.update_state(z, a)

        mig_val = mig_mod.compute()

        np.testing.assert_allclose(
            mig_val,
            mig(
                np.concatenate(zl, axis=0),
                np.concatenate(al, axis=0),
                reg_dim=[2, 3, 4],
            ),
        )

    def test_mig_noreg_discrete(self):
        mig_mod = MutualInformationGap(discrete=True)

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randint(16, size=(16, 3))

            zl.append(z)
            al.append(a)

            mig_mod.update_state(z, a)

        mig_val = mig_mod.compute()

        np.testing.assert_allclose(
            mig_val, mig(np.concatenate(zl, axis=0), np.concatenate(al, axis=0), discrete=True)
        )

    def test_mig_reg_discrete(self):
        mig_mod = MutualInformationGap(reg_dim=[2, 3, 4], discrete=True)

        zl = []
        al = []

        for _ in range(3):
            z = np.random.randn(16, 16)
            a = np.random.randint(16, size=(16, 3))

            zl.append(z)
            al.append(a)

            mig_mod.update_state(z, a)

        mig_val = mig_mod.compute()

        np.testing.assert_allclose(
            mig_val,
            mig(
                np.concatenate(zl, axis=0),
                np.concatenate(al, axis=0),
                reg_dim=[2, 3, 4],
                discrete=True
            ),
        )
