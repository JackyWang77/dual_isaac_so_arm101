# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments (skip when omni/Isaac Sim not available, e.g. offline extract_attention_offline.py).
try:
    from .tasks import *
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"SO_101.tasks not loaded ({e}). Gym envs will not be registered.")
# Register UI extensions.
try:
    from .ui_extension_example import *
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"SO_101.ui_extension_example not loaded ({e}).")
