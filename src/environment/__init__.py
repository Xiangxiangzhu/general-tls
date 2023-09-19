"""SUMO Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register


register(
    id="sumo-general-rl-v0",
    entry_point="src.environment.env:SumoEnvironment",
    kwargs={"single_agent": True},
)
