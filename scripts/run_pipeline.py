from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scene_risk.data.loader import NuScenesLoader
from scene_risk.forecasting.constant_velocity import ConstantVelocityForecaster
from scene_risk.pipeline.scene_pipeline import ScenePipeline
from scene_risk.risk.assessor import RiskAssessor
from scene_risk.risk.levels import RiskThresholds
from scene_risk.visualization.bev_renderer import BEVRenderer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    loader = NuScenesLoader(cfg.data.dataroot, cfg.data.version)
    forecaster = ConstantVelocityForecaster(cfg.forecaster.horizon_s, cfg.forecaster.dt)
    thresholds = RiskThresholds(**OmegaConf.to_container(cfg.risk, resolve=True))
    assessor = RiskAssessor(thresholds)
    renderer = BEVRenderer()

    pipeline = ScenePipeline(loader, forecaster, assessor, renderer)
    output_dir = Path(cfg.output_dir) / cfg.scene_token[:12]
    pipeline.run(cfg.scene_token, output_dir)
    print(f"Wrote frames and metrics.json to {output_dir}")


if __name__ == "__main__":
    main()
