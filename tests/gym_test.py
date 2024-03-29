import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import src


def test_api():
    env = gym.make(
        "sumo-general-rl-v0",
        num_seconds=100,
        use_gui=False,
        net_file="nets/single-intersection/single-intersection.net.xml",
        route_file="nets/single-intersection/single-intersection.rou.xml",
    )
    env.reset()
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


if __name__ == "__main__":
    test_api()
