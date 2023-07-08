from typing import List
import random
from math import pi, cos, sin

class Orientation:
    def __init__(self, x, y, z) -> None:
        # Stored as a point on the unit sphere
        self.x = x
        self.y = y
        self.z = z

    def rotateZ(self, angle):
        # Rotate around the z-axis
        return Orientation(
            x=self.x * cos(angle) - self.y * sin(angle),
            y=self.x * sin(angle) + self.y * cos(angle),
            z=self.z

class PositionRange:
    def __init__(self, lower_x, upper_x, lower_y, upper_y, lower_z, upper_z) -> None:
        self.lower_x = lower_x
        self.upper_x = upper_x
        self.lower_y = lower_y
        self.upper_y = upper_y
        self.lower_z = lower_z
        self.upper_z = upper_z

    def getRandomPosition(self):
        return (
            random.random() * (self.upper_x - self.lower_x) - self.lower_x,
            random.random() * (self.upper_y - self.lower_y) - self.lower_y,
            random.random() * (self.upper_z - self.lower_z) - self.lower_z,
        )

class Observation:
    def __init__(self, aruco_id, relative_distance, relative_rotation) -> None:
        self.aruco_id = aruco_id
        self.relative_distance = relative_distance
        self.relative_rotation = relative_rotation

class Particle:
    def __init__(self, position, orientation, weight) -> None:
        self.position = position
        self.weight = weight

class ArUco:
    def __init__(self, aruco_id, position, orientation) -> None:
        self.aruco_id = aruco_id
        self.position = position
        self.orientation = orientation

class ParticleSystem:
    def __init__(self, num_particles, position_range, arucos) -> None:
        self.num_particles = num_particles
        self.particles = []
        for _ in range(num_particles):
            self.particles.append(
                Particle(
                    position=position_range.getRandomPosition(),
                    orientation=self.getRandomPointOnUnitSphere(),
                    weight=1/num_particles
                )
            )
        self.aruco_map = {}
        for aruco in arucos:
            self.aruco_map[aruco.aruco_id] = (aruco.position, aruco.orientation)

    def updateWeightsUsingObservations(self, observations: List[Observation]):
        for observation in observations:
            for particle in self.particles:
                aruco_position, aruco_orientation = self.aruco_map[observation.aruco_id]
                particle_position = particle.position
                particle_orientation = particle.orientation
                likelihood = self.getLikelihood(
                    particle_position, particle_orientation, aruco_position, aruco_orientation, observation.relative_position, observation.relative_distance
                )
                particle.weight *= likelihood
    
    def normalizeWeights(self):
        total_weight = 0
        for particle in self.particles:
            total_weight += particle.weight
        for particle in self.particles:
            particle.weight /= total_weight

    def resample(self):
        new_particles = []
        for _ in range(self.num_particles):
            random_weight = random.random()
            for particle in self.particles:
                random_weight -= particle.weight
                if random_weight <= 0:
                    new_particles.append(particle)
                    break
        self.particles = new_particles


    def getLikelihood(self, particle_position, particle_orientation, aruco_position, aruco_orientation, relative_position, relative_rotation):
        # Writing this as a longer comment because I can't be asked to write docs
        # So there are a few variables here: the particle position/orientation(where we think we might be),
        # the observed relative distance and orientation of a given aruco marker(which we get from the camera),
        # and the known position of the aruco marker(which we know beforehand and load before we even start the program)

        # The aruco codes have 4 faces, so we rotate the observed orientation 4 times around the z-axis to get 4 possible observed positions
        rotated_orientations = self.getRotatedOrientations(relative_rotation)
        distance = self.getDistance(particle_position, estimated_position)
        return 1 / (distance + 1)

    def getRotatedOrientations(self, orientation):
        return (
            orientation.rotateZ(0),
            orientation.rotateZ(pi/2),
            orientation.rotateZ(pi),
            orientation.rotateZ(3*pi/2),
        )
    
    def getDistance(self, position1, position2):
        return (position1[0] - position2[0])**2 + (position1[1] - position2[1])**2 + (position1[2] - position2[2])**2
        
    def getRandomPointOnUnitSphere(self):
        x = random.random() * 2 - 1
        y = random.random() * 2 - 1
        z = random.random() * 2 - 1
        return (x, y, z)