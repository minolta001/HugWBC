
# from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# from .base.legged_robot import LeggedRobot
# from .h1.h1 import H1Robot
# from .h1.h1wbc import H1WBCRobot
# from .h1.h1mob import H1MOBRobot


# from legged_gym.envs.h1.h1_config import H1Cfg, H1CfgPPO
# from legged_gym.envs.h1.h1wbc_config import H1WBCCfg, H1WBCCfgPPO
# from legged_gym.utils.task_registry import task_registry
# from legged_gym.envs.h1.h1wbc_config import H1TSCfg, H1TSCfgPPO
# from legged_gym.envs.h1.h1mob_config import H1MOBCfg, H1MOBCfgPPO
# from legged_gym.envs.h1.h1mob_config import H1MOBTSCfg, H1MOBTSCfgPPO


# from legged_gym.envs.h1.h1interrupt import H1InterruptRobot
# from legged_gym.envs.h1.h1interrupt_config import H1InterruptCfg, H1InterruptCfgPPO
# # from legged_gym.envs.h1.h1interrupt_config import H1InterruptTSCfg, H1InterruptTSCfgPPO
# # from legged_gym.envs.h1.h1_evaluate_jump import H1EvaluateJumpRobot

# task_registry.register( "h1", H1Robot, H1Cfg(), H1CfgPPO())
# task_registry.register( "h1wbc", H1WBCRobot, H1WBCCfg(), H1WBCCfgPPO())
# task_registry.register( "h1ts", H1WBCRobot, H1TSCfg(), H1TSCfgPPO())
# task_registry.register( "h1mob", H1MOBRobot, H1MOBCfg(), H1MOBCfgPPO())
# task_registry.register( "h1mobts", H1MOBRobot, H1MOBTSCfg(), H1MOBTSCfgPPO())
# task_registry.register( "h1int", H1InterruptRobot, H1InterruptCfg(), H1InterruptCfgPPO())
# # task_registry.register( "h1intts", H1InterruptRobot, H1InterruptTSCfg(), H1InterruptTSCfgPPO())

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .h1.h1 import H1Robot
from legged_gym.envs.h1.h1interrupt import H1InterruptRobot

from legged_gym.envs.h1.h1_config import H1Cfg, H1CfgPPO
from legged_gym.envs.h1.h1interrupt_config import H1InterruptCfg, H1InterruptCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1", H1Robot, H1Cfg(), H1CfgPPO())
task_registry.register( "h1int", H1InterruptRobot, H1InterruptCfg(), H1InterruptCfgPPO())

