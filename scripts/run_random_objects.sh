#!/usr/bin/env bash
set -euo pipefail

readonly ISAAC_PYTHON="${HOME}/isaacsim/python.sh"
readonly SIMULATION_SCRIPT="${HOME}/Projects/dual_manipulation_isaac_sim/scripts/run_simulation_obj.py"

# Candidate pool passed to run_simulation_obj.py, which samples five unique models.
readonly OBJECTS=(
  apple
  banana
  foam_brick
  cracker_box
  hammer
  medium_clamp
  065-a_cups
  mug
  mustard_bottle
  pear
  power_drill
  phillips_screwdriver
  scissors
  strawberry
  tennis_ball
  tomato_soup_can
)

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "Isaac Sim Python launcher is not executable: ${ISAAC_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${SIMULATION_SCRIPT}" ]]; then
  echo "Simulation script does not exist: ${SIMULATION_SCRIPT}" >&2
  exit 1
fi

exec "${ISAAC_PYTHON}" "${SIMULATION_SCRIPT}" \
  --number 5 \
  --objects "${OBJECTS[@]}" \
  --x-range -0.3 0.3 \
  --y-range -0.3 0.0 \
  --control-mode mit \
  "$@"
