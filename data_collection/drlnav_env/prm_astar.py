import random, math

class PRM_utils:
    @classmethod
    def distance(cls, point1, point2, dimension):
        distance = 0
        for i in range(dimension):
            distance += (point1[i]-point2[i])**2
        return distance**0.5

    @classmethod
    def check_outside(cls, point, obs_list) -> bool:
        for obs in obs_list:
            distance = PRM_utils.distance(point, obs, 2)
            if distance <= obs[2]:
                return False
        return True

    @classmethod
    def check_not_intersect(cls, point1, point2, obs_list) -> bool:
        for obs in obs_list:
            vector1_2 = (point2[0]-point1[0], point2[1]-point1[1])
            vector1_0 = (obs[0]-point1[0], obs[1]-point1[1])
            distance_h = abs(vector1_2[0]*vector1_0[1]-vector1_2[1]*vector1_0[0])
            if distance_h > obs[2]:
                continue
            if vector1_0[0]*vector1_2[0]+vector1_0[1]*vector1_2[1] < 0:
                continue
            vector2_1 = (point1[0]-point2[0], point1[1]-point2[1])
            vector2_0 = (obs[0]-point2[0], obs[1]-point2[1])
            if vector2_0[0]*vector2_1[0]+vector2_0[1]*vector2_1[1] < 0:
                continue
            return False
        return True
    
def get_global_path(start_pose, end_pose, obs_list, shape, *args):
    # PRM
    n_sample = args[0]
    nearset_distance = args[1]
    map_size = args[2]
    inflation_radius = args[3]
    start_point = (start_pose[0], start_pose[1])
    end_point = (end_pose[0], end_pose[1])
    obs_list = [(obs[0], obs[1], obs[-1]+shape+inflation_radius) for obs in obs_list]


    sample_points = []
    while len(sample_points) < n_sample:
        # consider wall shape in our most maps.
        x, y = random.uniform(0, map_size[0]), random.uniform(0, map_size[1])
        if PRM_utils.check_outside((x, y), obs_list):
            sample_points.append((x, y))
    sample_points.append(start_point)
    sample_points.append(end_point)
    n_sample += 2

    distance_matrix = {}
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            distance_matrix[(i, j)] = PRM_utils.distance(sample_points[i], sample_points[j], 2)

    nearset_list = []
    for i in range(n_sample):
        nearset_i = []
        for j in range(n_sample):
            if j==i:continue
            m = 0
            while m < len(nearset_i):
                if distance_matrix.get((i, j), distance_matrix.get((j, i))) < distance_matrix.get((i, nearset_i[m]), distance_matrix.get((nearset_i[m], i))):break
                m += 1
            nearset_i.insert(m, j)
        nearset_list.append(nearset_i)

    graph = []
    for _ in range(n_sample):
        graph.append({})
    for i in range(n_sample):
        connect = 0
        for j in range(n_sample-1):
            p1, p2 = i, nearset_list[i][j]
            if PRM_utils.check_not_intersect(sample_points[p1], sample_points[p2], obs_list):
                connect += 1
                graph[p1][p2] = distance_matrix.get((p1,p2), distance_matrix.get((p2,p1)))
                graph[p2][p1] = distance_matrix.get((p1,p2), distance_matrix.get((p2,p1)))
            if connect >= nearset_distance:
                break

    # A*
    open = [n_sample-2]
    close = []
    points_marks = []
    for _ in range(n_sample):
        points_marks.append([-1, 2**32]) # father, cost
    points_marks[n_sample-2][1] = 0
    while len(open)!= 0:
        F_list = [points_marks[i][1]+distance_matrix.get((i, n_sample-1), 0) for i in open]
        p = open[F_list.index(min(F_list))]
        if p == n_sample-1:break
        open.remove(p)
        close.append(p)
        for child in graph[p].keys():
            if child not in close:
                if child not in open:
                    open.append(child)
                if points_marks[p][1]+graph[p][child] < points_marks[child][1]:
                    points_marks[child][1] = points_marks[p][1]+graph[p][child]
                    points_marks[child][0] = p
    
    global_path_tuple = []
    p = n_sample-1
    while p != -1:
        global_path_tuple.insert(0, p)
        p = points_marks[p][0]
    global_path_tuple = [sample_points[i] for i in global_path_tuple]
    
    part_path = []
    for i in range(5):
        if i < len(global_path_tuple) -1:
            yaw = math.atan2(global_path_tuple[i+1][1]-global_path_tuple[i][1], global_path_tuple[i+1][0]-global_path_tuple[i][0]) % (2*math.pi)
            part_path.append([global_path_tuple[i][0], global_path_tuple[i][1], yaw])
        else:
            part_path.append([global_path_tuple[-1][0], global_path_tuple[-1][1], end_pose[2]])

    return part_path
