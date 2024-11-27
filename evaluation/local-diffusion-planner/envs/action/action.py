from typing import List, Tuple


class Action:
    pass


class ContinuousAction(Action):
    v : float
    w : float
    beep: int

    def __init__(self, v, w, traj_v, traj_w, global_path_x, global_path_y, beep=0):
        self.v = v
        self.w = w
        self.traj_v = traj_v
        self.traj_w = traj_w
        self.global_path_x = global_path_x
        self.global_path_y = global_path_y

    def reverse(self):
        return [self.v, self.w, self.traj_v, self.traj_w, self.global_path_x, self.global_path_y]


class DiscreteActions:
    actions: List[ContinuousAction] = []

    def __init__(self, actions: List[Tuple]):
        for action in actions:
            assert action[0] >= 0
            assert len(action) == 2
            self.actions.append(ContinuousAction(*action))

    def __len__(self):
        return len(self.actions)
 
    def __getitem__(self, index):
        return self.actions[index]
