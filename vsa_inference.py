"""Small VSA representations and finite-state inference for closure experiments.

Only NumPy and SciPy are required.  The exact pixel observation model assumes
independent binary-pixel flips.  The VSA backend replaces its squared residual
with a compressed-vector residual: this is an approximate *pseudo-likelihood*,
not a calibrated probability model for image observations.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from scipy.special import logsumexp, xlogy


def log_normalize(log_weights, axis=-1):
    """Normalize log weights stably; reject NaNs, +inf, and empty support."""
    values = np.asarray(log_weights, dtype=np.float64)
    if values.size == 0 or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Log weights must be nonempty, with no NaN or +inf.")
    normalizer = logsumexp(values, axis=axis, keepdims=True)
    if not np.isfinite(normalizer).all():
        raise ValueError("Each normalized slice needs at least one finite weight.")
    return values - normalizer


def softmax(log_weights, axis=-1):
    """Return probabilities from unnormalized log weights."""
    return np.exp(log_normalize(log_weights, axis=axis))


def entropy(probabilities, axis=None):
    """Shannon entropy in nats, with 0 log 0 = 0; inputs are not renormalized."""
    q = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative.")
    return -np.sum(xlogy(q, q), axis=axis)


def _log_joint(loglik, logprior, strength):
    likelihood = np.asarray(loglik, dtype=np.float64)
    prior = np.asarray(logprior, dtype=np.float64)
    if likelihood.ndim != 1 or likelihood.shape != prior.shape:
        raise ValueError("loglik and logprior must be equal-length 1D arrays.")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative.")
    # A zero-strength ablation must not turn 0 * (-inf) into NaN.
    return likelihood.copy() if strength == 0 else likelihood + strength * prior


def posterior(loglik, logprior, strength=1):
    """Exact finite-state posterior for the supplied (possibly pseudo-)scores."""
    return softmax(_log_joint(loglik, logprior, strength))


def _cardinalities(cardinalities):
    cardinalities = tuple(cardinalities)
    cards = tuple(int(c) for c in cardinalities)
    if not cards or any(c <= 0 for c in cards):
        raise ValueError("cardinalities must contain positive integers.")
    if any(c != original for c, original in zip(cards, cardinalities)):
        raise ValueError("cardinalities must be integers.")
    return cards


def _states_array(states, cardinalities, require_complete=False):
    original = np.asarray(states)
    if original.ndim != 2 or original.shape[1] != len(cardinalities):
        raise ValueError("states must have shape (number_of_states, number_of_factors).")
    if not np.isfinite(original).all():
        raise ValueError("states must be finite integer indices.")
    result = original.astype(np.int64)
    if not np.equal(original, result).all():
        raise ValueError("states must contain integer indices.")
    if ((result < 0) | (result >= np.asarray(cardinalities))).any():
        raise ValueError("A state contains an out-of-range factor index.")
    if require_complete:
        count = int(np.prod(cardinalities, dtype=np.int64))
        if len(result) != count or len(np.unique(result, axis=0)) != count:
            raise ValueError("Mean-field inference requires the full Cartesian state set once.")
    return result


def factor_marginals(q, states, cardinalities):
    """Marginalize a distribution whose entries correspond to rows of states."""
    cards = _cardinalities(cardinalities)
    states = _states_array(states, cards)
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (len(states),) or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("q must contain one finite nonnegative weight per state.")
    return [
        np.bincount(states[:, i], weights=q, minlength=card)
        for i, card in enumerate(cards)
    ]


def mean_field(loglik, logprior, states, cardinalities, strength=1, sweeps=20, seed=0,
               message_vsa=None):
    """Update a categorical factorized approximation, with optional VSA messages.

    The states must enumerate the full Cartesian product, in any order.  Each
    default factor update exactly minimizes free energy given the other
    marginals.  If message_vsa is a compatible FactorVSA, expected bottom-up
    and top-down log-score messages are centered, encoded, and approximately
    decoded before the update.  VSA crosstalk can then increase the ORIGINAL
    free energy: this optional path is not exact coordinate descent.

    Finite joint log scores are required: smooth hard-zero priors before using
    this approximation.
    Arbitrary correlated hard support is not compatible with a full-support
    factorized initialization.

    Returns marginals, joint (the product approximation on the supplied rows),
    free_energy (initial value followed by one value per sweep), states, and
    final bottom_messages/top_messages (expected conditional log scores), and
    message_backend.  VSA runs also contain decoded_bottom_messages and
    decoded_top_messages for comparison with the original scalar messages.
    The trace need not reach the global optimum; use different seeds to inspect
    local optima.  This objective also works for pseudo-likelihood scores, but
    optimizing it does not make those scores calibrated.
    """
    cards = _cardinalities(cardinalities)
    states = _states_array(states, cards, require_complete=True)
    logjoint = _log_joint(loglik, logprior, strength)
    if logjoint.shape != (len(states),) or not np.isfinite(logjoint).all():
        raise ValueError("Mean-field inference needs one finite log score per state.")
    if int(sweeps) != sweeps or sweeps < 0:
        raise ValueError("sweeps must be a nonnegative integer.")
    if message_vsa is not None:
        if not isinstance(message_vsa, FactorVSA) or message_vsa.cardinalities != cards:
            raise ValueError("message_vsa must be a FactorVSA with matching cardinalities.")
    bottom_scores = np.asarray(loglik, dtype=np.float64)
    top_scores = (
        np.zeros_like(logjoint) if strength == 0
        else strength * np.asarray(logprior, dtype=np.float64)
    )
    rng = np.random.default_rng(seed)
    marginals = []
    for card in cards:
        initial = rng.uniform(0.25, 1.25, size=card)
        marginals.append(initial / initial.sum())

    def product_distribution():
        joint = np.ones(len(states), dtype=np.float64)
        for index, marginal in enumerate(marginals):
            joint *= marginal[states[:, index]]
        return joint

    def free_energy():
        joint = product_distribution()
        return float(-np.dot(joint, logjoint) - sum(entropy(q) for q in marginals))

    trace = [free_energy()]
    for _ in range(int(sweeps)):
        for index, card in enumerate(cards):
            # Never divide a joint weight by q_i: q_i can underflow to zero.
            other_weights = np.ones(len(states), dtype=np.float64)
            for other, marginal in enumerate(marginals):
                if other != index:
                    other_weights *= marginal[states[:, other]]
            if message_vsa is None:
                expected_log_score = np.bincount(
                    states[:, index], weights=other_weights * logjoint, minlength=card
                )
            else:
                bottom = np.bincount(
                    states[:, index], weights=other_weights * bottom_scores, minlength=card
                )
                top = np.bincount(
                    states[:, index], weights=other_weights * top_scores, minlength=card
                )
                expected_log_score = (
                    message_vsa.transmit_scores(bottom, index)
                    + message_vsa.transmit_scores(top, index)
                )
            marginals[index] = softmax(expected_log_score)
        trace.append(free_energy())

    bottom_messages, top_messages = [], []
    for index, card in enumerate(cards):
        other_weights = np.ones(len(states), dtype=np.float64)
        for other, marginal in enumerate(marginals):
            if other != index:
                other_weights *= marginal[states[:, other]]
        bottom_messages.append(np.bincount(
            states[:, index], weights=other_weights * bottom_scores, minlength=card
        ))
        top_messages.append(np.bincount(
            states[:, index], weights=other_weights * top_scores, minlength=card
        ))

    result = {
        "marginals": [q.copy() for q in marginals],
        "joint": product_distribution(),
        "free_energy": np.asarray(trace),
        "states": states.copy(),
        "bottom_messages": bottom_messages,
        "top_messages": top_messages,
        "message_backend": "dense" if message_vsa is None else "vsa",
    }
    if message_vsa is not None:
        result["decoded_bottom_messages"] = [
            message_vsa.transmit_scores(scores, index)
            for index, scores in enumerate(bottom_messages)
        ]
        result["decoded_top_messages"] = [
            message_vsa.transmit_scores(scores, index)
            for index, scores in enumerate(top_messages)
        ]
    return result


class PixelVSA:
    """Bind row, column, and pixel role codes; superpose weighted pixel codes.

    ``A`` has shape (height * width, dim), in image row-major order.  Each row
    has unit norm.  Consequently the encoded residual norm is comparable in
    expectation to the pixel residual norm, but finite-dimensional crosstalk
    changes individual scores.  This does not confer perceptual semantics.
    """

    def __init__(self, size, dim=512, seed=0):
        if np.isscalar(size):
            size = (int(size), int(size))
        else:
            size = tuple(int(s) for s in size)
        if len(size) != 2 or min(size) <= 0 or int(dim) != dim or dim <= 0:
            raise ValueError("size must specify positive height/width; dim must be positive.")
        self.size = size
        self.dim = int(dim)
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.x_keys = rng.choice([-1.0, 1.0], size=(size[1], self.dim)).astype(np.float32)
        self.y_keys = rng.choice([-1.0, 1.0], size=(size[0], self.dim)).astype(np.float32)
        self.role = rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        self.A = (
            self.y_keys[:, None, :] * self.x_keys[None, :, :] * self.role
        ).reshape(-1, self.dim) / np.float32(np.sqrt(self.dim))

    def _mask(self, mask):
        if mask is None:
            return np.ones(self.size, dtype=bool)
        result = np.asarray(mask, dtype=bool)
        if result.shape != self.size:
            raise ValueError(f"mask must have image shape {self.size}.")
        return result

    def encode(self, image, mask=None):
        """Encode one image or a batch with trailing shape (height, width)."""
        image = np.asarray(image, dtype=np.float32)
        if image.ndim < 2 or image.shape[-2:] != self.size:
            raise ValueError(f"image must have trailing shape {self.size}.")
        visible = self._mask(mask)
        # np.where also ensures unobserved NaNs cannot contaminate evidence.
        image = np.where(visible, image, 0.0)
        if not np.isfinite(image).all():
            raise ValueError("Visible image values must be finite.")
        flat = image.reshape(-1, np.prod(self.size))
        encoded = flat @ self.A
        return encoded.reshape(image.shape[:-2] + (self.dim,))

    def evidence_scores(self, images, observed, mask, flip_probability=0.05, backend="vsa"):
        """Score candidates using only visible pixels.

        For binary images the pixels backend is the exact independent-flip
        log likelihood up to a common additive constant.  For other numeric
        images it is simply a squared-error score.  The vsa backend always
        uses an approximate compressed-distance pseudo-likelihood.  Images
        have shape (K, H, W); no hidden truth or category label is consulted.
        """
        candidates = np.asarray(images, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float32)
        if candidates.ndim != 3 or candidates.shape[1:] != self.size:
            raise ValueError(f"images must have shape (K, {self.size[0]}, {self.size[1]}).")
        if observed.shape != self.size:
            raise ValueError(f"observed must have shape {self.size}.")
        if not 0 < flip_probability < 1:
            raise ValueError("flip_probability must lie strictly between 0 and 1.")
        if backend not in ("vsa", "pixels"):
            raise ValueError("backend must be 'vsa' or 'pixels'.")
        visible = self._mask(mask)
        beta = np.log1p(-flip_probability) - np.log(flip_probability)
        result = np.empty(len(candidates), dtype=np.float64)
        for start in range(0, len(candidates), 128):
            stop = min(start + 128, len(candidates))
            difference = np.where(visible, candidates[start:stop] - observed, 0.0)
            if not np.isfinite(difference).all():
                raise ValueError("Visible candidate and observation values must be finite.")
            flat = difference.reshape(stop - start, -1)
            residual = flat @ self.A if backend == "vsa" else flat
            squared_distance = np.einsum("ij,ij->i", residual, residual, dtype=np.float64)
            result[start:stop] = -beta * squared_distance
        return result


class FactorVSA:
    """Bind feature values inside role-coded groups, then superpose the groups.

    Codebooks are independent random bipolar vectors, not semantic embeddings.
    Groups map role names to sequences of factor indices.  The same named
    role code is reused within an instance.  Factor codebooks remain fixed
    during inference; inference selects an index, not an arbitrary new code.
    """

    def __init__(self, cardinalities, dim=512, seed=0):
        self.cardinalities = _cardinalities(cardinalities)
        if int(dim) != dim or dim <= 0:
            raise ValueError("dim must be a positive integer.")
        self.dim = int(dim)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.codebooks = [
            self._rng.choice([-1.0, 1.0], size=(card, self.dim)).astype(np.float32)
            for card in self.cardinalities
        ]
        # Dedicated, fixed channel keys for factor-level score messages.  They
        # are distinct from part/group roles and never carry a class label.
        self.message_roles = self._rng.choice(
            [-1.0, 1.0], size=(len(self.cardinalities), self.dim)
        ).astype(np.float32)
        self.roles = {}

    def transmit_scores(self, scores, factor_index):
        """Encode/decode one feature's centered candidate log-score message.

        For codebook C (candidate rows), the transmitted message is
        role * sum_k centered_score[k] * C[k].  After unbinding, dot/D cleanup
        yields (C C.T / D) @ centered_score, followed by centering again.
        This is approximate projection with crosstalk, NOT inverse-Gram
        decoding.  Orthogonal codewords give an exact centered roundtrip.

        Centering makes the result invariant to adding a common log constant.
        A role is actually bound and unbound, although this per-channel toy
        does not additionally superpose unrelated feature channels together.
        """
        if int(factor_index) != factor_index or not 0 <= factor_index < len(self.cardinalities):
            raise ValueError("Invalid factor index.")
        factor_index = int(factor_index)
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (self.cardinalities[factor_index],) or not np.isfinite(scores).all():
            raise ValueError("scores must contain one finite value per factor candidate.")
        centered = scores - scores.mean()
        codebook = self.codebooks[factor_index]
        role = self.message_roles[factor_index]
        encoded = role * (centered @ codebook)
        unbound = role * encoded
        decoded = (codebook @ unbound) / self.dim
        return decoded - decoded.mean()

    def _groups(self, groups):
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError("groups must map at least one role name to factor indices.")
        checked = {}
        for name, indices in groups.items():
            if not isinstance(name, str):
                raise ValueError("Group role names must be strings.")
            original = tuple(indices)
            indices = tuple(int(i) for i in original)
            if any(i != j for i, j in zip(indices, original)):
                raise ValueError("Group indices must be integers.")
            if len(set(indices)) != len(indices):
                raise ValueError("A group must not repeat a factor (it would cancel).")
            if any(i < 0 or i >= len(self.cardinalities) for i in indices):
                raise ValueError("Group contains an invalid factor index.")
            checked[name] = indices
        for name in sorted(checked):
            if name not in self.roles:
                self.roles[name] = self._rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        return checked

    def _state(self, state):
        return _states_array(np.asarray(state)[None, :], self.cardinalities)[0]

    def encode(self, state, groups):
        """Encode a discrete state using a dictionary of role-defined groups."""
        state = self._state(state)
        groups = self._groups(groups)
        result = np.zeros(self.dim, dtype=np.float32)
        for name, indices in groups.items():
            term = self.roles[name].copy()
            for index in indices:
                term *= self.codebooks[index][state[index]]
            result += term
        return result

    def unbind_factor(self, encoded, state_estimate, factor_index, group_name, groups):
        """Unbind other estimated factors; other groups remain as crosstalk."""
        groups = self._groups(groups)
        state = self._state(state_estimate)
        if group_name not in groups or factor_index not in groups[group_name]:
            raise ValueError("Requested factor must belong to the requested group.")
        residual = np.asarray(encoded, dtype=np.float32).copy()
        if residual.shape != (self.dim,) or not np.isfinite(residual).all():
            raise ValueError("encoded must be one finite vector of length dim.")
        residual *= self.roles[group_name]
        for index in groups[group_name]:
            if index != factor_index:
                residual *= self.codebooks[index][state[index]]
        return residual

    def cleanup_factor(self, residual, factor_index):
        """Choose the codebook entry with greatest cosine similarity."""
        if not 0 <= factor_index < len(self.cardinalities):
            raise ValueError("Invalid factor index.")
        residual = np.asarray(residual, dtype=np.float64)
        if residual.shape != (self.dim,) or not np.isfinite(residual).all():
            raise ValueError("residual must be one finite vector of length dim.")
        # All codewords have equal norm; the common cosine denominator cancels.
        # A zero query provides no evidence and ties deterministically at index 0.
        return int(np.argmax(self.codebooks[factor_index] @ residual))

    cleanup = cleanup_factor
