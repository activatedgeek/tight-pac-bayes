import unittest
from pactl.bounds.get_pac_bounds import compute_catoni_bound
from pactl.bounds.get_pac_bounds import pac_bayes_bound_opt
from pactl.bounds.get_pac_bounds import compute_mcallester_bound


class TestBounds(unittest.TestCase):

    def test_bound_equiv(self):
        train_error = 1. - 0.710693
        # divergence = 1700
        divergence = 1724
        sample_size = int(3 * 1.e4)
        bound = compute_catoni_bound(train_error, divergence, sample_size)
        bound2 = pac_bayes_bound_opt(divergence, train_error, sample_size)
        bound3 = compute_mcallester_bound(train_error, divergence, sample_size)
        print('\nTEST: Bound Equiv')
        print(f'Train error: {train_error:1.3e}')
        print(f'Bound Catoni: {bound:1.3e}')
        print(f'Bound Zhou: {bound2:1.3e}')
        print(f'Bound Mc: {bound3:1.3e}')
        self.assertTrue(expr=bound >= bound2)


if __name__ == '__main__':
    unittest.main()
