import torch
from tqdm.auto import tqdm

from toma import toma

from .utils import gather_expand, batch_multi_choices


class JointEntropy:
    """Random variables (all with the same # of categories $C$) can be added via `JointEntropy.add_variables`.

    `JointEntropy.compute` computes the joint entropy.

    `JointEntropy.compute_batch` computes the joint entropy of the added variables with each of the variables in the provided batch probabilities in turn."""

    def compute(self) -> torch.Tensor:
        """Computes the entropy of this joint entropy."""
        raise NotImplementedError()

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "JointEntropy":
        """Expands the joint entropy to include more terms."""
        raise NotImplementedError()

    def compute_batch(
        self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None
    ) -> torch.Tensor:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        raise NotImplementedError()


class ExactJointEntropy(JointEntropy):
    joint_probs_M_K: torch.Tensor

    def __init__(self, joint_probs_M_K: torch.Tensor):
        self.joint_probs_M_K = joint_probs_M_K

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> "ExactJointEntropy":
        return ExactJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    def compute(self) -> torch.Tensor:
        probs_M = torch.mean(self.joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(probs_M) * probs_M
        entropy = torch.sum(nats_M)
        return entropy

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "ExactJointEntropy":
        assert self.joint_probs_M_K.shape[1] == log_probs_N_K_C.shape[1]

        N, K, C = log_probs_N_K_C.shape
        joint_probs_K_M_1 = self.joint_probs_M_K.t()[:, :, None]

        probs_N_K_C = log_probs_N_K_C.exp()

        # Using lots of memory.
        for i in range(N):
            probs_i__K_1_C = probs_N_K_C[i][:, None, :].to(
                joint_probs_K_M_1, non_blocking=True
            )
            joint_probs_K_M_C = joint_probs_K_M_1 * probs_i__K_1_C
            joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))

        self.joint_probs_M_K = joint_probs_K_M_1.squeeze(2).t()
        return self

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None):
        assert self.joint_probs_M_K.shape[1] == log_probs_B_K_C.shape[1]

        B, K, C = log_probs_B_K_C.shape
        M = self.joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(
                B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device
            )

        pbar = tqdm(total=B, desc="ExactJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)
        def chunked_joint_entropy(
            chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int
        ):
            chunked_probs_b_K_C = chunked_log_probs_b_K_C.exp()
            b = chunked_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.joint_probs_M_K.dtype,
                device=self.joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    self.joint_probs_M_K,
                    chunked_probs_b_K_C[i].to(self.joint_probs_M_K, non_blocking=True),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K

            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C, dim=(1, 2)),
                non_blocking=True,
            )

            pbar.update(end - start)

        pbar.close()

        return output_entropies_B


class SampledJointEntropy(JointEntropy):
    """Random variables (all with the same # of categories $C$) can be added via `SampledJointEntropy.add_variables`.

    `SampledJointEntropy.compute` computes the joint entropy.

    `SampledJointEntropy.compute_batch` computes the joint entropy of the added variables with each of the variables in the provided batch probabilities in turn."""

    sampled_joint_probs_M_K: torch.Tensor

    def __init__(
        self, sampled_joint_probs_M_K: torch.Tensor, scaling_factor: torch.Tensor = None
    ):
        if scaling_factor is not None:
            self.scaling_factor = scaling_factor
            self.sampled_joint_probs_M_K = sampled_joint_probs_M_K
        else:
            self.scaling_factor = self._calculate_scaling_factor(
                sampled_joint_probs_M_K
            )
            self.sampled_joint_probs_M_K = sampled_joint_probs_M_K / self.scaling_factor

    @staticmethod
    def _calculate_scaling_factor(
        sampled_joint_probs_M_K: torch.Tensor,
    ) -> torch.Tensor:
        scaling_factor = 1.0
        scaled_probs = sampled_joint_probs_M_K.clone()
        while scaled_probs.max() < 1e-2:
            scaling_factor /= 10
            scaled_probs *= 10
        return torch.Tensor([scaling_factor]).to(sampled_joint_probs_M_K)

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> "SampledJointEntropy":
        return SampledJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    @staticmethod
    def sample(probs_N_K_C: torch.Tensor, M: int) -> "SampledJointEntropy":
        K = probs_N_K_C.shape[1]

        # S: num of samples per w
        S = M // K

        # Samples depending on each "sample" of one observation:
        # e.g. have 2 samples of 1 obs with probas [[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]
        # Most probable sample output will be [0, 1]. Consequently, the first column of
        # `samples_M_K` will depend on [0.6, 0.3], while the second - on [0.1, 0.8]
        choices_N_K_S = batch_multi_choices(probs_N_K_C, S).long()

        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]

        probs_N_K_K_S = gather_expand(
            expanded_probs_N_K_1_C, dim=-1, index=expanded_choices_N_1_K_S
        )
        # exp sum log seems necessary to avoid 0s?
        log_sum_K_K_S = torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False)
        scaling_factor = log_sum_K_K_S.max()
        probs_K_K_S = torch.exp(log_sum_K_K_S - scaling_factor)
        samples_K_M = probs_K_K_S.reshape((K, -1))

        samples_M_K = samples_K_M.t()
        return SampledJointEntropy(samples_M_K, scaling_factor.exp())

    def compute(self) -> torch.Tensor:
        sampled_joint_probs_M = torch.mean(
            self.sampled_joint_probs_M_K, dim=1, keepdim=False
        )
        nats_M = -(torch.log(sampled_joint_probs_M) + self.scaling_factor.log())
        entropy = torch.mean(nats_M)
        return entropy

    def add_variables(
        self, log_probs_N_K_C: torch.Tensor, M2: int
    ) -> "SampledJointEntropy":
        K = log_probs_N_K_C.shape[1]
        assert self.sampled_joint_probs_M_K.shape[1] == K

        sample_K_M1_1 = self.sampled_joint_probs_M_K.t()[:, :, None]

        new_sample_M2_K = self.sample(log_probs_N_K_C.exp(), M2).sampled_joint_probs_M_K
        new_sample_K_1_M2 = new_sample_M2_K.t()[:, None, :]

        merged_sample_K_M1_M2 = sample_K_M1_1 * new_sample_K_1_M2
        merged_sample_K_M = merged_sample_K_M1_M2.reshape((K, -1))

        self.sampled_joint_probs_M_K = merged_sample_K_M.t()

        return self

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None):
        assert self.sampled_joint_probs_M_K.shape[1] == log_probs_B_K_C.shape[1]

        B, K, C = log_probs_B_K_C.shape
        M = self.sampled_joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(
                B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device
            )

        pbar = tqdm(total=B, desc="SampledJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)
        def chunked_joint_entropy(
            chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int
        ):
            b = chunked_log_probs_b_K_C.shape[0]
            chunked_probs_b_K_C = chunked_log_probs_b_K_C.exp()

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.sampled_joint_probs_M_K.dtype,
                device=self.sampled_joint_probs_M_K.device,
            )
            torch.matmul(
                self.sampled_joint_probs_M_K,
                chunked_probs_b_K_C.to(self.sampled_joint_probs_M_K, non_blocking=True),
                out=probs_b_M_C,
            )
            probs_b_M_C /= K

            q_1_M_1 = self.sampled_joint_probs_M_K.mean(dim=1, keepdim=True)[None]

            output_entropies_B[start:end].copy_(
                torch.sum(
                    -(torch.log(probs_b_M_C) + self.scaling_factor.log())
                    * probs_b_M_C
                    / q_1_M_1,
                    dim=(1, 2),
                )
                / M,
                non_blocking=True,
            )

            pbar.update(end - start)

        pbar.close()

        return output_entropies_B


class DynamicJointEntropy(JointEntropy):
    inner: JointEntropy
    log_probs_max_N_K_C: torch.Tensor
    N: int
    M: int

    def __init__(self, M: int, max_N: int, K: int, C: int, dtype=None, device=None):
        self.M = M
        self.N = 0
        self.max_N = max_N

        self.inner = ExactJointEntropy.empty(K, dtype=dtype, device=device)
        self.log_probs_max_N_K_C = torch.empty(
            (max_N, K, C), dtype=dtype, device=device
        )

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "DynamicJointEntropy":
        C = self.log_probs_max_N_K_C.shape[2]
        add_N = log_probs_N_K_C.shape[0]

        assert self.log_probs_max_N_K_C.shape[0] >= self.N + add_N
        assert self.log_probs_max_N_K_C.shape[2] == C

        self.log_probs_max_N_K_C[self.N : self.N + add_N] = log_probs_N_K_C
        self.N += add_N

        num_exact_samples = C**self.N
        if num_exact_samples > self.M:
            self.inner = SampledJointEntropy.sample(
                self.log_probs_max_N_K_C[: self.N].exp(), self.M
            )
        else:
            self.inner.add_variables(log_probs_N_K_C)

        return self

    def compute(self) -> torch.Tensor:
        return self.inner.compute()

    def compute_batch(
        self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None
    ) -> torch.Tensor:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        return self.inner.compute_batch(log_probs_B_K_C, output_entropies_B)
