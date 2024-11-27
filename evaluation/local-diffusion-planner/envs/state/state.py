import numpy as np


class ImageState():
    def __init__(self,
                 vector_states,
                 sensor_maps,
                 global_maps,
                 is_collisions,
                 is_arrives,
                 lasers,
                 ped_vector_states,
                 ped_maps,
                 refresh_num_episode,
                 run_dis_episode,
                 run_trajectory_points_episode_x,
                 run_trajectory_points_episode_y,
                 step_ds,
                 ped_min_dists,
                 velocity_a,
                 target_pose,
                 pose,
                 target_pose_visual,
                 pose_visual,
                 obs_list
                 ):
        assert len(vector_states) == len(sensor_maps) == len(is_collisions) == len(is_arrives) == len(lasers) \
                == len(ped_vector_states) == len(ped_maps) == len(refresh_num_episode) == len(run_dis_episode) \
                == len(run_trajectory_points_episode_x) == len(run_trajectory_points_episode_y) == len(step_ds) == len(ped_min_dists)

        self.vector_states = vector_states
        self.sensor_maps = sensor_maps
        self.global_maps = global_maps
        self.is_collisions = is_collisions
        self.is_arrives = is_arrives
        self.lasers = lasers
        self.ped_vector_states = ped_vector_states
        self.ped_maps = ped_maps
        self.ped_min_dists = ped_min_dists
        self.refresh_num_episode = refresh_num_episode
        self.run_dis_episode = run_dis_episode
        self.run_trajectory_points_episode_x = run_trajectory_points_episode_x
        self.run_trajectory_points_episode_y = run_trajectory_points_episode_y
        self.step_ds = step_ds
        self.velocity_a = velocity_a
        self.target_pose = target_pose
        self.pose = pose
        self.target_pose_visual = target_pose_visual
        self.pose_visual = pose_visual
        self.obs_list = obs_list
        

        # self.step_ds = np.zeros_like(len(self.vector_states), dtype=np.float)

    def __str__(self):
        return """Image State Info:
        vector_states: {}
        sensor_maps: {}
        is_collisions: {}
        is_arrives: {}
        lasers: {}
        ped_vector_states: {} 
        ped_maps: {}
        ped_min_dists: {}
        refresh_num_episode: {}
        run_dis_episode: {}
        run_trajectory_points_episode_x: {}
        run_trajectory_points_episode_y: {}
        step distance: {}
        velocity_a:{}
        target_pose:{}
        pose : {}
        """.format(self.vector_states,
                   self.sensor_maps,
                   self.is_collisions,
                   self.is_arrives,
                   self.lasers,
                   self.ped_vector_states,
                   self.ped_maps,
                   self.ped_min_dists,
                   self.refresh_num_episode,
                   self.run_dis_episode,
                   self.run_trajectory_points_episode_x,
                   self.run_trajectory_points_episode_y,
                   self.velocity_a,
                   self.step_ds,
                   self.target_pose,
                   self.pose)


    def get_sensor_maps(self):
        return self.sensor_maps

    def change_sensor_maps(self, sensor_state):
        self.sensor_maps = sensor_state

    def __len__(self):
        return len(self.vector_states)