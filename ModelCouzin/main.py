import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist


class World:
    def __init__(self, dimension, radius):
        self.dimension = dimension
        self.radius = radius


class Swarm:
    def __init__(self, swarm_number, dimension):
        self.number = swarm_number
        self.agent_list = list()
        self.state_position_dimension = np.zeros(shape=(swarm_number, dimension))
        self.state_direction_dimension = np.zeros(shape=(swarm_number, dimension))
        self.state_velocity = np.zeros(shape=swarm_number)
        self.matrix_distance = np.zeros(shape=(swarm_number, swarm_number))


class Agent:
    def __init__(self, name, dimension, perception_radius, perception_angle):
        # 个体运动基础属性
        self.name = name
        self.position_dimension = np.zeros(shape=dimension)
        self.direction_dimension = np.zeros(shape=dimension)
        self.velocity = 0
        # 个体感知范围属性
        self.perception_radius_repulsion = perception_radius[0]
        self.perception_radius_align = self.perception_radius_repulsion + perception_radius[1]
        self.perception_radius_attract = self.perception_radius_align + perception_radius[2]
        self.perception_angle = perception_angle
        # 相互作用属性
        self.neighbor_repulsion = list()
        self.neighbor_align = list()
        self.neighbor_attract = list()


def add_loop_boundary(agent_position_dimension, world_radius):
    for dimension in range(len(agent_position_dimension)):
        if agent_position_dimension[dimension] > world_radius:
            agent_position_dimension[dimension] = agent_position_dimension[dimension] - 2 * world_radius
        elif agent_position_dimension[dimension] < - world_radius:
            agent_position_dimension[dimension] = agent_position_dimension[dimension] + 2 * world_radius
    return agent_position_dimension


def display(swarm):
    display_body = swarm.state_position_dimension.T
    display_head = (swarm.state_position_dimension + swarm.state_direction_dimension).T
    plt.ion()  # 打开交互模式
    plt.cla()  # 清空画布
    plt.plot(display_body[0], display_body[1], 'r.')
    plt.plot(display_head[0], display_head[1], 'b.')
    plt.xlim(-world_couzin.radius, world_couzin.radius)
    plt.ylim(-world_couzin.radius, world_couzin.radius)
    for agent in swarm.agent_list:
        plt.text(agent.position_dimension[0], agent.position_dimension[1],
                 str(len(agent.neighbor_repulsion)+len(agent.neighbor_align)+len(agent.neighbor_attract)))
    plt.pause(0.000001)  # 暂停


def update_swarm(swarm):
    for swarm_agent in swarm.agent_list:
        swarm_agent.position_dimension = (swarm_agent.position_dimension + swarm_agent.direction_dimension *
                                          swarm_agent.velocity * init_arguments['world_time_step'])
        swarm_agent.position_dimension = add_loop_boundary(swarm_agent.position_dimension, world_couzin.radius)
        agent_direction = (((math.atan2(swarm_agent.direction_dimension[1], swarm_agent.direction_dimension[0])
                             * 180 / math.pi) + np.random.normal(loc=0, scale=init_arguments['agent_angle_error'],
                                                                 size=1)) * math.pi / 180)
        swarm_agent.direction_dimension[0] = math.cos(agent_direction)
        swarm_agent.direction_dimension[1] = math.sin(agent_direction)
        swarm_agent.velocity = swarm_agent.velocity
        # 更新swarm状态
        swarm.state_position_dimension[swarm_agent.name] = swarm_agent.position_dimension
        swarm.state_direction_dimension[swarm_agent.name] = swarm_agent.direction_dimension
        swarm.state_velocity[swarm_agent.name] = swarm_agent.velocity
    # 计算两点间距离
    swarm.matrix_distance = squareform(pdist(swarm.state_position_dimension))
    return swarm


def calculate_agent_neighbor(swarm):
    for agent_i in swarm.agent_list:
        agent_i.neighbor_repulsion.clear()
        agent_i.neighbor_align.clear()
        agent_i.neighbor_attract.clear()
        for agent_j in swarm.agent_list:
            if agent_i.name != agent_j.name:
                if swarm.matrix_distance[agent_i.name][agent_j.name] <= agent_i.perception_radius_repulsion:
                    agent_i.neighbor_repulsion.append(agent_j)
                elif swarm.matrix_distance[agent_i.name][agent_j.name] <= agent_i.perception_radius_align:
                    agent_i.neighbor_align.append(agent_j)
                elif swarm.matrix_distance[agent_i.name][agent_j.name] <= agent_i.perception_radius_attract:
                    agent_i.neighbor_attract.append(agent_j)


def init_swarm(world, swarm):
    for number in range(swarm.number):
        # 生成新agent
        agent = Agent(number, world.dimension, init_arguments['agent_perception_radius'],
                      init_arguments['agent_perception_angle'])
        agent.position_dimension = (np.random.random_sample(size=world.dimension) -
                                    0.5) * 2 * world.radius
        direction = (np.random.random_sample() - 0.5) * 2 * math.pi
        agent.direction_dimension[0] = math.cos(direction)
        agent.direction_dimension[1] = math.sin(direction)
        agent.velocity = init_arguments['agent_velocity']
        # 将agent添加到swarm，更新swarm状态
        swarm.agent_list.append(agent)
        swarm.state_position_dimension[agent.name] = agent.position_dimension
        swarm.state_direction_dimension[agent.name] = agent.direction_dimension
        swarm.state_velocity[agent.name] = agent.velocity
    # 计算两点间距离
    swarm.matrix_distance = squareform(pdist(swarm.state_position_dimension))


def calculate_itoj_distance_angle_2d(agent_i, agent_j):
    itoj_distance_dimension = np.zeros(shape=2)
    itoj_angle_dimension = np.zeros(shape=2)
    itoj_distance_dimension[0] = agent_j.position_dimension[0] - agent_i.position_dimension[0]
    itoj_distance_dimension[1] = agent_j.position_dimension[1] - agent_i.position_dimension[1]
    itoj_distance = math.sqrt(itoj_distance_dimension[0]**2 + itoj_distance_dimension[1]**2)
    itoj_angle = math.atan2(itoj_distance_dimension[1], itoj_distance_dimension[0])
    itoj_angle_dimension[0] = math.cos(itoj_angle)
    itoj_angle_dimension[1] = math.sin(itoj_angle)
    return itoj_distance, itoj_angle_dimension


if __name__ == '__main__':
    init_arguments = {'world_dimension': 2,
                      'world_radius': 50,
                      'swarm_number': 2,
                      'agent_velocity': 10,
                      'world_time_step': 0.1,
                      'agent_angle_error': 0.01,
                      'agent_perception_radius': (10, 20, 20),
                      'agent_perception_angle': 270
    }
    world_couzin = World(init_arguments['world_dimension'], init_arguments['world_radius'])
    swarm_couzin = Swarm(init_arguments['swarm_number'], world_couzin.dimension)
    init_swarm(world_couzin, swarm_couzin)
    ticks = 0
    while True:
        # 绘图
        display(swarm_couzin)
        print(ticks)
        ticks = ticks + 1
        # 更新swarm
        swarm_couzin = update_swarm(swarm_couzin)
        # 计算邻居
        calculate_agent_neighbor(swarm_couzin)

        # 计算个体方向
        # 问题：位置更新使用当前方向还是之后方向？
