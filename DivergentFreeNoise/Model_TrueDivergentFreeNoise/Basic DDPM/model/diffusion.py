from __future__ import annotations

from dataclasses import dataclass

import torch

from model_parameters.loss_functions import masked_mse, masked_mse_with_mask
from model_parameters.noise_schedules import DiffusionSchedule, make_schedule
from model_parameters.noise_types import build_noise_sampler
from model.pathing import build_inpainting_condition


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    batch = timesteps.shape[0]
    extracted = values.gather(0, timesteps)
    return extracted.reshape(batch, *((1,) * (len(target_shape) - 1)))


@dataclass
class GaussianDiffusion:
    timesteps: int = 1000
    noise_type: str = "divergence_free"
    prediction_type: str = "x0"
    reconstruction_loss_weight: float = 1.0
    noise_loss_weight: float = 0.25

    def __post_init__(self) -> None:
        self.schedule: DiffusionSchedule | None = None
        self.noise_sampler = build_noise_sampler(self.noise_type)
        self.prediction_type = self.prediction_type.strip().lower().replace("-", "_")
        if self.prediction_type not in {"epsilon", "x0"}:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

    def to(self, device: torch.device) -> "GaussianDiffusion":
        self.schedule = make_schedule(self.timesteps, device=device)
        return self

    def _require_schedule(self) -> DiffusionSchedule:
        if self.schedule is None:
            self.schedule = make_schedule(self.timesteps)
        return self.schedule

    def _sample_noise(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.noise_sampler.sample(shape, device=device, dtype=dtype)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
        if mask.shape[0] == 1 and reference.shape[0] > 1:
            mask = mask.expand(reference.shape[0], -1, -1, -1)
        return mask.to(device=reference.device, dtype=reference.dtype)

    def _model_predict_x0(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_output = model(x, timesteps, conditioning=conditioning)
        if self.prediction_type == "epsilon":
            schedule = self._require_schedule()
            sqrt_alphas_cumprod = _extract(schedule.sqrt_alphas_cumprod, timesteps, x.shape)
            sqrt_one_minus = _extract(schedule.sqrt_one_minus_alphas_cumprod, timesteps, x.shape)
            predicted_x0 = (x - sqrt_one_minus * model_output) / torch.clamp(sqrt_alphas_cumprod, min=1e-20)
        else:
            predicted_x0 = model_output
        return torch.nan_to_num(predicted_x0, nan=0.0, posinf=0.0, neginf=0.0)

    def _predict_noise_from_x0(self, x: torch.Tensor, timesteps: torch.Tensor, predicted_x0: torch.Tensor) -> torch.Tensor:
        schedule = self._require_schedule()
        sqrt_alphas_cumprod = _extract(schedule.sqrt_alphas_cumprod, timesteps, x.shape)
        sqrt_one_minus = _extract(schedule.sqrt_one_minus_alphas_cumprod, timesteps, x.shape)
        return (x - sqrt_alphas_cumprod * predicted_x0) / torch.clamp(sqrt_one_minus, min=1e-20)

    def _reconstruction_mask(
        self,
        land_mask: torch.Tensor,
        observation_mask: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        ocean = (~land_mask.bool()).to(device=reference.device)
        ocean = self._expand_mask(ocean, reference)
        if observation_mask is None:
            return ocean
        known = self._expand_mask(observation_mask, reference)
        return ocean * (1.0 - known)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        schedule = self._require_schedule()
        if noise is None:
            noise = self._sample_noise(tuple(x_start.shape), x_start.device, x_start.dtype)
        sqrt_alphas_cumprod = _extract(schedule.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = _extract(schedule.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise

    def training_loss(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        land_mask: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = x_start.shape[0]
        device = x_start.device
        timesteps = torch.randint(0, self.timesteps, (batch,), device=device, dtype=torch.long)
        noise = self._sample_noise(tuple(x_start.shape), device, x_start.dtype)
        x_noisy = self.q_sample(x_start, timesteps, noise=noise)
        predicted_x0 = self._model_predict_x0(model, x_noisy, timesteps, conditioning=conditioning)
        predicted_noise = self._predict_noise_from_x0(x_noisy, timesteps, predicted_x0)
        noise_loss = masked_mse(predicted_noise, noise, land_mask)
        reconstruction_mask = self._reconstruction_mask(land_mask, observation_mask, x_start)
        reconstruction_loss = masked_mse_with_mask(predicted_x0, x_start, reconstruction_mask)
        return self.noise_loss_weight * noise_loss + self.reconstruction_loss_weight * reconstruction_loss

    def p_mean_variance(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        schedule = self._require_schedule()
        betas = _extract(schedule.betas, timesteps, x.shape)
        alphas = _extract(schedule.alphas, timesteps, x.shape)
        alphas_cumprod = _extract(schedule.alphas_cumprod, timesteps, x.shape)
        alphas_cumprod_prev = _extract(schedule.alphas_cumprod_prev, timesteps, x.shape)
        predicted_x0 = self._model_predict_x0(model, x, timesteps, conditioning=conditioning)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
        posterior_mean_coef2 = torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
        model_mean = posterior_mean_coef1 * predicted_x0 + posterior_mean_coef2 * x
        posterior_variance = _extract(schedule.posterior_variance, timesteps, x.shape)
        return model_mean, posterior_variance

    def p_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        land_mask: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_mean, posterior_variance = self.p_mean_variance(model, x, timesteps, conditioning=conditioning)
        if torch.all(timesteps == 0):
            sample = model_mean
        else:
            noise = self._sample_noise(tuple(x.shape), x.device, x.dtype)
            sample = model_mean + torch.sqrt(torch.clamp(posterior_variance, min=1e-20)) * noise
        sample = torch.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
        ocean_mask = (~land_mask.bool()).to(device=x.device, dtype=x.dtype)
        if ocean_mask.dim() == 2:
            ocean_mask = ocean_mask.unsqueeze(0).unsqueeze(0)
        elif ocean_mask.dim() == 3:
            ocean_mask = ocean_mask.unsqueeze(1)
        return sample * ocean_mask

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...],
        land_mask: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_schedule()
        device = land_mask.device
        x = self._sample_noise(shape, device, torch.float32)
        ocean_mask = (~land_mask.bool()).to(device=device, dtype=x.dtype)
        if ocean_mask.dim() == 2:
            ocean_mask = ocean_mask.unsqueeze(0).unsqueeze(0)
        elif ocean_mask.dim() == 3:
            ocean_mask = ocean_mask.unsqueeze(1)
        x = x * ocean_mask
        for timestep in reversed(range(self.timesteps)):
            timesteps = torch.full((shape[0],), timestep, device=device, dtype=torch.long)
            x = self.p_sample(model, x, timesteps, land_mask, conditioning=conditioning)
        return x

    @torch.no_grad()
    def repaint(
        self,
        model: torch.nn.Module,
        observed: torch.Tensor,
        observation_mask: torch.Tensor,
        land_mask: torch.Tensor,
        resample: int = 10,
        skip_last_step: bool = False,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = observed.device
        if conditioning is None and getattr(model, "condition_channels", 0) > 0:
            conditioning = build_inpainting_condition(observed, observation_mask, land_mask)
        x = self._sample_noise(tuple(observed.shape), device, observed.dtype)
        ocean_mask = (~land_mask.bool()).to(device=device, dtype=observed.dtype)
        if ocean_mask.dim() == 2:
            ocean_mask = ocean_mask.unsqueeze(0).unsqueeze(0)
        elif ocean_mask.dim() == 3:
            ocean_mask = ocean_mask.unsqueeze(1)
        x = x * ocean_mask
        if observation_mask.dim() == 2:
            known = observation_mask.unsqueeze(0).unsqueeze(0)
        elif observation_mask.dim() == 3:
            known = observation_mask.unsqueeze(1)
        else:
            known = observation_mask
        known = known.to(device=device, dtype=observed.dtype)

        start_timestep = self.timesteps - 1
        if skip_last_step:
            start_timestep = max(1, self.timesteps - 1)

        for timestep in reversed(range(start_timestep + 1)):
            if skip_last_step and timestep == 0:
                continue
            timesteps = torch.full((observed.shape[0],), timestep, device=device, dtype=torch.long)
            for _ in range(max(1, resample)):
                x = self.p_sample(model, x, timesteps, land_mask, conditioning=conditioning)
                x = known * observed + (1.0 - known) * x
                x = x * ocean_mask
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
