from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan
from linear_operator.operators import DiagLinearOperator
import torch


class MultitaskFixedNoiseGaussianLikelihood(MultitaskGaussianLikelihood):
    def __init__(
        self,
        num_tasks,
        independent_noise,
        noise_constraint=None,
        noise=0.0001,
        **kwargs,
    ):
        # Ensure independent noise is provided
        if independent_noise is None:
            raise ValueError("independent_noise must be provided")

        # Initialize the parent class with fixed global noise
        super().__init__(
            num_tasks, has_global_noise=True, has_task_noise=False, rank=0, **kwargs
        )

        # Set the global noise to a constant value
        self.noise = noise

        # Set up independent noise
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)
        self.register_parameter(
            name="raw_independent_noise", parameter=torch.nn.Parameter(torch.zeros(1))
        )
        self.register_constraint("raw_independent_noise", noise_constraint)
        self._set_independent_noise(independent_noise)

    def _set_independent_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if value.ndim == 0:
            value = value.unsqueeze(0)
        transformed_value = self.raw_independent_noise_constraint.inverse_transform(
            value
        )
        self.raw_independent_noise.data = transformed_value

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        # noise_covar = super()._shaped_noise_covar(base_shape, *params, **kwargs) - include if wanting a global noise term

        independent_noise = self.raw_independent_noise_constraint.transform(
            self.raw_independent_noise
        )

        if independent_noise.shape[0] < base_shape[-2]:
            last_noise = independent_noise[-1]
            independent_noise = torch.cat(
                [
                    independent_noise,
                    last_noise.expand(base_shape[-2] - independent_noise.shape[0]),
                ]
            )
        elif independent_noise.shape[0] > base_shape[-2]:
            independent_noise = independent_noise[: base_shape[-2]]

        independent_noise = independent_noise.repeat_interleave(self.num_tasks)
        independent_noise_covar = DiagLinearOperator(independent_noise)
        noise_covar = independent_noise_covar

        return noise_covar
