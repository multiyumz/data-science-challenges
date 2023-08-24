from nbresult import ChallengeResultTestCase


class TestSolution(ChallengeResultTestCase):
    def test_cv_results(self):
        self.assertEqual(self.result.scoring, "precision")
        self.assertGreaterEqual(self.result.cv, 2)
        self.assertGreaterEqual(len(self.result.mean_test_score), 5)
