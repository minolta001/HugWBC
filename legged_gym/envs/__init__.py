from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# Import H1 V1 classes
from .h1.h1 import H1Robot
from .h1.h1interrupt import H1InterruptRobot
from .h1.h1_config import H1Cfg, H1CfgPPO
from .h1.h1interrupt_config import H1InterruptCfg, H1InterruptCfgPPO

# Import H1 V2 classes
from .h1_2.h1_2 import H12Robot
from .h1_2.h1_2_config import H12Cfg, H12CfgPPO
from .h1_2.h1_2interrupt import H12InterruptRobot
from .h1_2.h1_2interrupt_config import H12InterruptCfg, H12InterruptCfgPPO

# Imporat H1 V2 FPO classes
#from .h1_2_fpo.h1_2 import H12Robot
#from .h1_2_fpo.h1_2_config import H12Cfg, H12CfgPPO
from .h1_2_fpo.h1_2_fpo_interrupt import H12_FPO_InterruptRobot
from .h1_2_fpo.h1_2_fpo_interrupt_config import H12_FPO_InterruptCfg, H12_FPO_InterruptCfgFPO


from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1int", H1InterruptRobot, H1InterruptCfg(), H1InterruptCfgPPO())

# Register the new H1 V2 interrupt environment
task_registry.register( "h1_2int", H12InterruptRobot, H12InterruptCfg(), H12InterruptCfgPPO())

# Register H1 V2 FPO interrupt environment
task_registry.register( "h1_2_fpo_int", H12_FPO_InterruptRobot, H12_FPO_InterruptCfg(), H12_FPO_InterruptCfgFPO())