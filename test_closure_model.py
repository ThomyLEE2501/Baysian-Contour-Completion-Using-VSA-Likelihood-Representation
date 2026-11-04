"""Small numerical and information-leakage checks for the closure sandbox.

Run from this directory with: python -m unittest -v test_closure_model
These checks verify implementation invariants, not psychological validity.
"""

import itertools
import unittest

import numpy as np

from closure_model import (
    CARDINALITIES,
    CLASS_NAMES,
    PRIOR_NAMES,
    completion_metrics,
    fit_priors,
    make_library,
    observe,
    sample_training,
    sample_trials,
)
from vsa_inference import (
    FactorVSA,
    PixelVSA,
    factor_marginals,
    mean_field,
    posterior,
    softmax,
)


class LibraryAndPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = make_library(size=24, seed=7)
        cls.training_ids, cls.labels = sample_training(cls.library, n_per_class=20, seed=11)
        cls.priors = fit_priors(cls.library, cls.training_ids, cls.labels)

    def test_renderer_has_binary_pixels_and_complete_factor_vocabulary(self):
        self.assertEqual(len(self.library.states), int(np.prod(CARDINALITIES)))
        self.assertEqual(self.library.images.shape, (len(self.library.states), 24, 24))
        self.assertTrue(np.isin(self.library.images, [0, 1]).all())
        self.assertEqual(len(np.unique(self.library.states, axis=0)), len(self.library.states))

    def test_training_and_evaluation_combinations_are_disjoint(self):
        self.assertEqual(len(np.intersect1d(self.library.train_ids, self.library.test_ids)), 0)
        self.assertTrue(np.isin(self.training_ids, self.library.train_ids).all())
        self.assertFalse(np.isin(self.training_ids, self.library.test_ids).any())
        self.assertEqual(len(self.training_ids), 20 * len(CLASS_NAMES))
        trials = sample_trials(self.library, n_per_visibility=3, visibilities=(.2,), seed=2)
        self.assertTrue(all(t["index"] in self.library.test_ids for t in trials))

    def test_fit_priors_rejects_test_state_as_training_example(self):
        with self.assertRaises(ValueError):
            fit_priors(self.library, np.array([self.library.test_ids[0]]), np.array([0]))

    def test_priors_are_normalized_and_cover_the_same_candidates(self):
        self.assertEqual(set(self.priors.logpriors), set(PRIOR_NAMES))
        for name, logprior in self.priors.logpriors.items():
            with self.subTest(prior=name):
                self.assertEqual(logprior.shape, (len(self.library.states),))
                self.assertTrue(np.isfinite(logprior).all())
                self.assertAlmostEqual(float(np.exp(logprior).sum()), 1.0)

    def test_generic_prior_ignores_morphology_given_layout_and_contour(self):
        states = self.library.states
        generic = self.priors.logpriors["generic"]
        for layout in range(CARDINALITIES[4]):
            for contour in range(CARDINALITIES[5]):
                selected = generic[(states[:, 4] == layout) & (states[:, 5] == contour)]
                np.testing.assert_allclose(selected, selected[0], atol=1e-12)

    def test_prior_fitting_does_not_read_generator_probabilities(self):
        changed = make_library(size=24, seed=7)
        changed.teacher_logp[:] = -123.0
        new_priors = fit_priors(changed, self.training_ids, self.labels)
        for name in PRIOR_NAMES:
            np.testing.assert_array_equal(new_priors.logpriors[name], self.priors.logpriors[name])

    def test_observation_zeros_unseen_pixels(self):
        truth = self.library.images[0]
        mask = np.zeros_like(truth, dtype=bool)
        mask[:8] = True
        result = observe(truth, mask, flip_probability=0, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(result[mask], truth[mask])
        self.assertTrue(np.all(result[~mask] == 0))

    def test_metrics_distinguish_missing_and_invented_contours(self):
        truth = np.array([[1, 0], [0, 0]], dtype=float)
        observed = np.zeros_like(truth)
        mask = np.zeros_like(truth, dtype=bool)
        correct = completion_metrics(truth, truth, observed, mask)
        all_ink = completion_metrics(np.ones_like(truth), truth, observed, mask)
        no_ink = completion_metrics(np.zeros_like(truth), truth, observed, mask)
        self.assertEqual(correct["hidden_brier"], 0)
        self.assertEqual(correct["hidden_recall"], 1)
        self.assertEqual(correct["hidden_false_positive"], 0)
        self.assertEqual(all_ink["hidden_recall"], 1)
        self.assertEqual(all_ink["hidden_false_positive"], 1)
        self.assertEqual(no_ink["hidden_recall"], 0)
        self.assertEqual(no_ink["hidden_false_positive"], 0)


class FactorBindingTests(unittest.TestCase):
    def test_single_group_unbinding_recovers_each_factor(self):
        encoder = FactorVSA((3, 2, 3), dim=256, seed=11)
        state = np.array([2, 1, 0])
        groups = {"nose": (0, 1, 2)}
        encoded = encoder.encode(state, groups)
        for j in range(3):
            with self.subTest(factor=j):
                residual = encoder.unbind_factor(encoded, state, j, "nose", groups)
                np.testing.assert_array_equal(residual, encoder.codebooks[j][state[j]])
                self.assertEqual(encoder.cleanup_factor(residual, j), int(state[j]))

    def test_score_transmission_is_invariant_to_a_common_log_offset(self):
        encoder = FactorVSA((3,), dim=64, seed=12)
        scores = np.array([0.2, -2.0, 1.3])
        base = encoder.transmit_scores(scores, 0)
        shifted = encoder.transmit_scores(scores + 1000.0, 0)
        np.testing.assert_allclose(shifted, base, atol=1e-10, rtol=0)

    def test_orthogonal_codewords_give_exact_centered_score_roundtrip(self):
        encoder = FactorVSA((3,), dim=4, seed=12)
        encoder.codebooks[0] = np.array([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
        ], dtype=np.float32)
        scores = np.array([-1.7, 0.4, 2.8])
        actual = encoder.transmit_scores(scores, 0)
        np.testing.assert_allclose(actual, scores - scores.mean(), atol=1e-12, rtol=0)


class PixelEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.encoder = PixelVSA(size=4, dim=256, seed=71)
        self.images = np.zeros((4, 4, 4), dtype=np.float32)
        self.images[0, 0, :] = 1
        self.images[1, 1, :] = 1
        self.images[2, 2, :] = 1
        self.images[3, :, 3] = 1

    def test_full_observation_recovers_matching_candidate(self):
        mask = np.ones((4, 4), dtype=bool)
        loglik = self.encoder.evidence_scores(
            self.images, self.images[2], mask, backend="pixels"
        )
        q = posterior(loglik, np.zeros(4))
        self.assertEqual(int(np.argmax(q)), 2)
        self.assertGreater(q[2], 0.999)
        self.assertAlmostEqual(float(q.sum()), 1.0)

    def test_exact_pixel_score_is_bernoulli_log_likelihood_up_to_constant(self):
        mask = np.ones((4, 4), dtype=bool)
        flip_probability = 0.13
        loglik = self.encoder.evidence_scores(
            self.images, self.images[0], mask,
            flip_probability=flip_probability, backend="pixels"
        )
        mismatches = np.sum(self.images != self.images[0], axis=(1, 2))
        expected = -np.log((1 - flip_probability) / flip_probability) * mismatches
        np.testing.assert_allclose(loglik - loglik[0], expected, atol=1e-5)

    def test_hidden_observation_values_never_affect_evidence(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2] = True
        observed_a = self.images[0].copy()
        observed_b = observed_a.copy()
        observed_b[~mask] = 1 - observed_b[~mask]
        for backend in ("pixels", "vsa"):
            with self.subTest(backend=backend):
                a = self.encoder.evidence_scores(self.images, observed_a, mask, backend=backend)
                b = self.encoder.evidence_scores(self.images, observed_b, mask, backend=backend)
                np.testing.assert_allclose(a, b, atol=1e-6)

    def test_hidden_candidate_values_never_affect_evidence(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0] = True
        changed = self.images.copy()
        changed[:, ~mask] = 1 - changed[:, ~mask]
        for backend in ("pixels", "vsa"):
            with self.subTest(backend=backend):
                a = self.encoder.evidence_scores(self.images, self.images[0], mask, backend=backend)
                b = self.encoder.evidence_scores(changed, self.images[0], mask, backend=backend)
                np.testing.assert_allclose(a, b, atol=1e-6)

    def test_no_observed_pixels_leaves_only_the_prior(self):
        mask = np.zeros((4, 4), dtype=bool)
        prior = np.array([0.1, 0.2, 0.3, 0.4])
        for backend in ("pixels", "vsa"):
            with self.subTest(backend=backend):
                loglik = self.encoder.evidence_scores(
                    self.images, self.images[0], mask, backend=backend
                )
                q = posterior(loglik, np.log(prior))
                np.testing.assert_allclose(q, prior, atol=1e-7)

    def test_vsa_encoding_is_linear(self):
        a, b = self.images[0], self.images[1]
        np.testing.assert_allclose(
            self.encoder.encode(a + b),
            self.encoder.encode(a) + self.encoder.encode(b),
            atol=1e-6,
        )

    def test_seed_reproduces_pixel_codebook(self):
        same = PixelVSA(size=4, dim=256, seed=71)
        different = PixelVSA(size=4, dim=256, seed=72)
        np.testing.assert_array_equal(self.encoder.encode(self.images[0]), same.encode(self.images[0]))
        self.assertFalse(np.array_equal(self.encoder.encode(self.images[0]), different.encode(self.images[0])))


class PosteriorTests(unittest.TestCase):
    def setUp(self):
        self.cardinalities = (2, 3)
        self.states = np.array(list(itertools.product(range(2), range(3))))

    def test_stable_softmax(self):
        q = softmax(np.array([10000.0, 10001.0, 9999.0]))
        self.assertTrue(np.isfinite(q).all())
        self.assertAlmostEqual(float(q.sum()), 1.0)
        self.assertEqual(int(np.argmax(q)), 1)

    def test_zero_prior_strength_discards_prior(self):
        loglik = np.array([-2.0, 0.0, -1.0])
        q = posterior(loglik, np.array([-100.0, -50.0, 0.0]), strength=0)
        np.testing.assert_allclose(q, softmax(loglik), atol=1e-12)

    def test_marginals_are_normalized(self):
        q = softmax(np.arange(len(self.states), dtype=float))
        marginals = factor_marginals(q, self.states, self.cardinalities)
        for j, marginal in enumerate(marginals):
            self.assertEqual(len(marginal), self.cardinalities[j])
            self.assertAlmostEqual(float(np.sum(marginal)), 1.0)

    def test_mean_field_matches_exact_for_separable_distribution(self):
        first = np.array([0.6, -0.9])
        second = np.array([-1.0, 0.5, 0.2])
        loglik = first[self.states[:, 0]] + second[self.states[:, 1]]
        logprior = np.zeros(len(self.states))
        exact = posterior(loglik, logprior)
        result = mean_field(
            loglik, logprior, self.states, self.cardinalities,
            sweeps=8, seed=2,
        )
        np.testing.assert_allclose(result["joint"], exact, atol=1e-8)

    def test_orthogonal_vsa_messages_match_separable_exact_posterior(self):
        encoder = FactorVSA(self.cardinalities, dim=4, seed=42)
        hadamard = np.array([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ], dtype=np.float32)
        encoder.codebooks[0] = hadamard[[1, 2]]
        encoder.codebooks[1] = hadamard[[0, 1, 3]]
        loglik = np.array([0.6, -0.9])[self.states[:, 0]] + np.array([-1.0, 0.5, 0.2])[self.states[:, 1]]
        logprior = np.array([-0.3, 0.7])[self.states[:, 0]] + np.array([1.1, -0.1, 0.4])[self.states[:, 1]]
        exact = posterior(loglik, logprior, strength=0.8)
        result = mean_field(
            loglik, logprior, self.states, self.cardinalities,
            strength=0.8, sweeps=8, seed=2, message_vsa=encoder,
        )
        self.assertEqual(result["message_backend"], "vsa")
        np.testing.assert_allclose(result["joint"], exact, atol=1e-10, rtol=0)

    def test_mean_field_coordinate_updates_do_not_increase_free_energy(self):
        rng = np.random.default_rng(99)
        loglik = rng.normal(size=len(self.states))
        logprior = rng.normal(size=len(self.states))
        result = mean_field(
            loglik, logprior, self.states, self.cardinalities,
            strength=0.8, sweeps=15, seed=3,
        )
        trace = np.asarray(result["free_energy"])
        self.assertTrue(np.isfinite(trace).all())
        self.assertTrue(np.all(np.diff(trace) <= 1e-9), msg=f"Free-energy trace: {trace}")
        self.assertAlmostEqual(float(np.sum(result["joint"])), 1.0)


if __name__ == "__main__":
    unittest.main()
