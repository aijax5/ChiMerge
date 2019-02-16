from discretization.chi_merge import *

import unittest
import time

import numpy as np
import pandas as pd
from pandas.util import testing as pdt
import multiprocessing as mp

from sklearn.datasets import load_iris

class TestChiMerge(unittest.TestCase):
    # TODO: parent class for preparing test data
    def load_iris_for_here(self):
        data = load_iris()
        features = pd.DataFrame(data['data'], columns=['a', 'b', 'c', 'd'])
        target = pd.Series(data=data['target'], name='target')

        return features, target

    def test_original_should_not_be_changed(self):
        data = load_iris()
        features = pd.DataFrame(data['data'], columns=['a', 'b', 'c', 'd'])
        target = pd.Series(data=data['target'], name='target')
        features_copy = features.copy()

        chi_merge = ChiMerge(con_features=features.columns, significance_level=0.1)
        chi_merge.fit_transform(X=features, y=target)

        pdt.assert_frame_equal(features_copy, features)

    def test_cutpoints_according_to_sigificance_level(self):
        '''
        higher significance level should have many cutpoints that lower significance level
        :return:
        '''
        features, target = self.load_iris_for_here()

        high_sig = ChiMerge(con_features=features.columns, significance_level=0.5)
        low_sig = ChiMerge(con_features=features.columns, significance_level=0.1)

        high_sig.fit(features, target)
        low_sig.fit(features, target)

        high_sig_cutpoints = [len(cutpoints) for cutpoints in high_sig.cutpoints.values()]
        low_sig_cutpoints = [len(cutpoints) for cutpoints in low_sig.cutpoints.values()]

        self.assertGreaterEqual(high_sig_cutpoints, low_sig_cutpoints)

    def test_parallel_iris_small_features(self):
        data = load_iris()
        features = pd.DataFrame(data['data'], columns=['a', 'b', 'c', 'd'])
        target = pd.Series(data=data['target'], name='target')

        single_mdlp = ChiMerge(con_features=features.columns, significance_level=0.1)
        single_dis = single_mdlp.fit_transform(X=features, y=target)

        parallel_mdlp4 = ChiMerge(con_features=features.columns, significance_level=0.1, n_jobs=4)
        parallel_dis4 = parallel_mdlp4.fit_transform(X=features, y=target)

        parallel_mdlp10 = ChiMerge(con_features=features.columns, significance_level=0.1, n_jobs=10)
        parallel_dis10 = parallel_mdlp10.fit_transform(X=features, y=target)

        self.assertEqual(single_mdlp.cutpoints, parallel_mdlp4.cutpoints)
        self.assertEqual(parallel_mdlp4.cutpoints, parallel_mdlp10.cutpoints)
        pdt.assert_frame_equal(single_dis, parallel_dis4)
        pdt.assert_frame_equal(parallel_dis4, parallel_dis10)

        # TODO: if max_cutpoints is applied, its time should be usually longer


if __name__ == '__main__':
    unittest.main()