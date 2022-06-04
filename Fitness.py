# P19129 Paraskevi Palioura

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    # Function to calculate route distance from given graph
    def route_distance(self, graph):
        if self.distance == 0:
            path_distance = 0
            for i in range(0, len(self.route)):
                from_city = self.route[i]
                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    to_city = self.route[0]
                path_distance += graph[int(from_city) - 1, int(to_city) - 1]
            self.distance = path_distance
        return self.distance

    def route_fitness(self, graph):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance(graph))
        return self.fitness
