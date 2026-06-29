from __future__ import annotations

import pytest

from scene_risk.data.extractor import _map_category
from scene_risk.data.schemas import AgentCategory


@pytest.mark.parametrize(
    ("category_name", "expected"),
    [
        ("vehicle.bicycle", AgentCategory.CYCLIST),
        ("vehicle.motorcycle", AgentCategory.CYCLIST),
        ("vehicle.car", AgentCategory.VEHICLE),
        ("vehicle.truck", AgentCategory.VEHICLE),
        ("vehicle.bus.rigid", AgentCategory.VEHICLE),
        ("human.pedestrian.adult", AgentCategory.PEDESTRIAN),
        ("human.pedestrian.child", AgentCategory.PEDESTRIAN),
        ("movable_object.barrier", AgentCategory.OTHER),
        ("static_object.bicycle_rack", AgentCategory.OTHER),
        ("animal", AgentCategory.OTHER),
    ],
)
def test_map_category(category_name: str, expected: AgentCategory) -> None:
    assert _map_category(category_name) == expected
