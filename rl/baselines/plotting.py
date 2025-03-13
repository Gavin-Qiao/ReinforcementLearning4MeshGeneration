from stable_baselines3.common import results_plotter
from pathlib import Path
import configparser

base_path = Path(__file__).parent.parent.parent
config = configparser.ConfigParser()
config.read(f'{base_path}/config')

version = 2

results_plotter.plot_results([f"{config['default']['a2c_log']}/{version}/"],
                             1e5, results_plotter.X_TIMESTEPS, "TD3 LunarLander")