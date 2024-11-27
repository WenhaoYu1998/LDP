import gym

from envs.state import ImageState


class ObsStateTmp(gym.ObservationWrapper):
    def __init__(self, env, cfg):
        super(ObsStateTmp, self).__init__(env)

    def observation(self, states: ImageState):

        return [states.sensor_maps, states.vector_states, states.target_pose, states.pose, states.obs_list, states.global_maps, states.target_pose_visual, states.pose_visual, states.robot_list]

