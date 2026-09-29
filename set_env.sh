# Session setup. Source after activating the environment:
#
#   conda activate needlework && source set_env.sh
#
# NEEDLEWORK_ROOT is the one project variable. Set it once (for example in ~/.bashrc)
# to a directory on a disk with room for datasets and checkpoints. Everything else is
# derived from it by needlework.paths:
#
#   $NEEDLEWORK_ROOT/data     downloaded inputs (robomimic HDF5, prepared zarr stores)
#   $NEEDLEWORK_ROOT/outputs  everything a run writes (training runs, stitched stores)
#   $NEEDLEWORK_ROOT/cache    re-creatable assets (DINOv3 source and weights)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file instead of running it: source set_env.sh" >&2
    exit 1
fi

if [[ -z "${NEEDLEWORK_ROOT:-}" ]]; then
    echo "NEEDLEWORK_ROOT is not set. Example: export NEEDLEWORK_ROOT=/scratch/\$USER/needlework" >&2
    return 1
fi
if [[ "${NEEDLEWORK_ROOT}" != /* ]]; then
    echo "NEEDLEWORK_ROOT must be an absolute path: ${NEEDLEWORK_ROOT}" >&2
    return 1
fi
mkdir -p "${NEEDLEWORK_ROOT}"/{data,outputs,cache} || return 1

# Headless MuJoCo rendering for robomimic rollouts. The GPU used for rendering is chosen
# by the code from the training device, never set here.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
