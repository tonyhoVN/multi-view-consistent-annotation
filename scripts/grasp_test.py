from execute_grasp import execute_grasp_plan, load_grasp_plan, print_plan
from pathlib import Path

EXECUTE_GRASP = True
GRASP_CALL_TIMEOUT = 120.0
grasp_output_path = Path("/home/hier-tony/Projects/vla_dual_arm/scan_output/grasp_commands/hammer.json")

# Reload and validate the serialized file that will drive RobotAPI.
validated_grasp_plan = load_grasp_plan(grasp_output_path, maximum_opening_m = 0.15)
print_plan(validated_grasp_plan)

# Keep hardware motion explicitly opt-in for safe notebook debugging.
if EXECUTE_GRASP:
    execute_grasp_plan(
        validated_grasp_plan, call_timeout=GRASP_CALL_TIMEOUT, use_sim_time=True
    )
    print("Grasp command sequence completed")
else:
    print("Dry run only. Set EXECUTE_GRASP = True to move the robot.")



