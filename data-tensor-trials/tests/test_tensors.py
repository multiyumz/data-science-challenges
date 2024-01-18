from nbresult import ChallengeResultTestCase
import tensorflow as tf
import numpy as np

class TestTensors(ChallengeResultTestCase):

    def assertTensorsEqual(self, tensor1, tensor2):
        np.testing.assert_equal(tensor1.numpy(), tensor2.numpy())

    def test_a(self):
        expected_a = tf.ones(shape=(3, 3))
        self.assertTensorsEqual(self.result.a, expected_a)

    def test_b(self):
        expected_b = tf.expand_dims(tf.ones(shape=(3, 3)), 0)
        self.assertTensorsEqual(self.result.b, expected_b)

    def test_c(self):
        expected_c = tf.zeros(shape=(9, 1))
        self.assertTensorsEqual(self.result.c, expected_c)

    def test_d_shape(self):
        expected_d_shape = (1, 3, 3)
        self.assertEqual(self.result.d_shape, expected_d_shape)

    def test_e(self):
        expected_e = tf.constant([[[0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]]])
        self.assertTensorsEqual(self.result.e, expected_e)

    def test_f(self):
        expected_f = tf.constant([[[1.0, 2.0, 3.0],
                                   [1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0]]])
        self.assertTensorsEqual(self.result.f, expected_f)

    def test_g(self):
        expected_g = tf.constant([3.0, 3.0, 6.0, 9.0])
        self.assertTensorsEqual(self.result.g, expected_g)

    def test_h(self):
        expected_h = tf.constant([[[0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0]]])
        self.assertTensorsEqual(self.result.h, expected_h)

    def test_i(self):
        expected_i = tf.constant([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        self.assertTensorsEqual(self.result.i, expected_i)

    def test_j(self):
        expected_j = tf.constant([1, 2, 3, 4, 8])
        self.assertTensorsEqual(self.result.j, expected_j)

    def test_k(self):
        expected_k = tf.constant([[1, 2, 3, 4, 8]])
        self.assertTensorsEqual(self.result.k, expected_k)

    def test_l(self):
        expected_k = tf.constant([[1, 2, 3, 4, 8]])
        expected_l = tf.tile(expected_k, (50, 1))
        self.assertTensorsEqual(self.result.l, expected_l)

    def test_m(self):
        expected_k = tf.constant([[1, 2, 3, 4, 8]])
        expected_l = tf.tile(expected_k, (50, 1))
        expected_m = expected_l != 3
        self.assertTensorsEqual(self.result.m, expected_m)

    def test_n(self):
        expected_k = tf.constant([[1, 2, 3, 4, 8]])
        expected_l = tf.tile(expected_k, (50, 1))
        expected_n = expected_l /3
        self.assertTensorsEqual(self.result.n, expected_n)

    def test_o_shape(self):
        expected_o_shape = [10, 5]
        self.assertEqual(self.result.o_shape, expected_o_shape)
