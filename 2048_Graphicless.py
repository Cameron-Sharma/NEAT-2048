import random
import math
import os


def add_num(board):
    zeroes = False

    for i in range(4):
        if not all(board[i]):
            zeroes = True

    if zeroes:
        if random.randint(0, 9) == 9:
            num = 4
        else:
            num = 2

        found = False

        while not found:
            pos = random.randint(0, 15)

            if board[pos // 4][pos % 4] == 0:
                board[pos // 4][pos % 4] = num
                found = True

    return board


class Functions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-1 * x))

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def ReLU(x):
        if x > 0:
            return x
        else:
            return 0

    @staticmethod
    def leaky_ReLU(x):
        if x > 0:
            return x
        else:
            return 0.01 * x

    @staticmethod
    def get_parametric_ReLU(a):
        def parametric_ReLU(x):
            if x > 0:
                return x
            else:
                return a * x

        return parametric_ReLU

    @staticmethod
    def sinusoid(x):
        return math.sin(x)

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def heaviside_step(x):
        if x < 0:
            return 0
        else:
            return 1


class Layer:
    def __init__(self, activation_func):
        self.activation_func = activation_func
        self.weights = None

    def rand_weights(self, num_inputs, num_outputs):
        self.weights = []

        for i in range(num_outputs):
            self.weights.append([])

            for j in range(num_inputs + 1):
                self.weights[i].append(random.random())

    def calculate(self, inputs):
        outputs = []

        for weight_set in self.weights:
            output = 0

            for i in range(len(weight_set)):
                if i == len(weight_set) - 1:
                    output += weight_set[i]
                else:
                    output += inputs[i] * weight_set[i]

            outputs.append(self.activation_func(output))

        return outputs


class Individual:
    keys = ["U", "D", "L", "R"]

    def __init__(self, layers):
        self.layers = layers
        self.keystroke = 0

    def calculate(self, board):
        inputs = []
        for row in board:
            inputs.extend(row)

        for layer in self.layers:
            inputs = layer.calculate(inputs)

        indices = [0]

        for i in range(1, len(inputs)):
            for j, index in enumerate(indices):
                if inputs[index] < inputs[i]:
                    indices.insert(j, i)
                    break
            else:
                indices.append(i)

        return self.keys[indices[self.keystroke]]

    def mate(self, parent2):
        if len(self.layers) != len(parent2.layers):
            raise ValueError("There must be the same number of layers")

        layers = []

        for self_layer, parent2_layer in zip(self.layers, parent2.layers):
            if len(self_layer.weights) != len(parent2_layer.weights):
                raise ValueError("Weights must be the same size")

            if self_layer.activation_func != parent2_layer.activation_func:
                raise ValueError("Layers must have the same activation function")

            layer = Layer(self_layer.activation_func)

            weights = []

            for self_weight_set, parent2_weight_set in zip(self_layer.weights, parent2_layer.weights):
                if len(self_weight_set) != len(parent2_weight_set):
                    raise ValueError("Weights must be the same size")

                weights.append([])

                for i in range(len(self_weight_set)):
                    probability = random.randint(0, 99)

                    if probability < 45:
                        weights[len(weights) - 1].append(self_weight_set[i])
                    elif probability < 90:
                        weights[len(weights) - 1].append(parent2_weight_set[i])
                    else:
                        weights[len(weights) - 1].append(random.random())

            layer.weights = weights
            layers.append(layer)

        return Individual(layers)


def NEAT():
    num_generations = 10000
    num_indivs = 100
    layer_sizes = ((16, 8), (8, 4))
    activation_funcs = [Functions.sigmoid, Functions.leaky_ReLU]
    num_trials = 5

    print("Generation 1")

    try:
        os.mkdir("Example_Learning/Generation_1")
    except FileExistsError:
        filepath = "Example_Learning/Generation_1"

        if os.listdir(filepath):
            clear = input("Need to delete data from {}. Type \"clear\" if you want to clear: ".format(filepath))

            if clear == "clear":
                for file in os.listdir(filepath):
                    os.remove(os.path.join(filepath, file))

    individuals = {}

    for i in range(num_indivs):
        layers = []

        for j in range(len(layer_sizes)):
            layer = Layer(activation_funcs[j])
            layer.rand_weights(layer_sizes[j][0], layer_sizes[j][1])
            layers.append(layer)

        individual = Individual(layers)

        total_score = 0

        for k in range(num_trials):
            total_score += simulate_2048(individual)

        individuals[individual] = total_score / num_trials

        with open("Example_Learning/Generation_1/{}_Number_{}.txt".format(individuals[individual], i), "w") as fh:
            output_text = ""

            for layer in layers:
                function_name = str(layer.activation_func)

                while function_name.find(".") != -1:
                    function_name = function_name[function_name.find(".") + 1:]

                function_name = function_name[:function_name.find(" at")]

                output_text += "Activation function: {}\nWeights: {}\n\n".format(function_name, layer.weights)

            fh.write(output_text)

    all_values = individuals.values()

    total = 0
    count = 0

    for value in all_values:
        total += value
        count += 1

    print("Average:", total / count)

    for generation in range(2, num_generations + 1):
        sorted_indivs = sorted(individuals, key=lambda x: -1 * individuals[x])

        new_generation = []

        new_generation.extend(sorted_indivs[:int(0.1 * len(sorted_indivs))])

        potential_parents = sorted_indivs[:int(0.5 * len(sorted_indivs))]

        for i in range(len(sorted_indivs) - len(new_generation)):
            parent1 = random.choice(potential_parents)
            parent2 = random.choice(potential_parents)
            new_generation.append(parent1.mate(parent2))

        print("Generation", generation)

        try:
            os.mkdir("Example_Learning/Generation_" + str(generation))
        except FileExistsError:
            filepath = "Example_Learning/Generation_" + str(generation)

            if os.listdir(filepath):
                clear = input("Need to delete data from {}. Type \"clear\" if you want to clear: ".format(filepath))

                if clear == "clear":
                    for file in os.listdir(filepath):
                        os.remove(os.path.join(filepath, file))

        individuals = {}

        count = 0

        for individual in new_generation:
            total_score = 0

            for k in range(num_trials):
                total_score += simulate_2048(individual)

            individuals[individual] = total_score / num_trials

            with open("Example_Learning/Generation_{}/{}_Number_{}.txt".format(generation, individuals[individual], count),
                      "w") as fh:
                output_text = ""

                for layer in individual.layers:
                    function_name = str(layer.activation_func)

                    while function_name.find(".") != -1:
                        function_name = function_name[function_name.find(".") + 1:]

                    function_name = function_name[:function_name.find(" at")]

                    output_text += "Activation function: {}\nWeights: {}\n\n".format(function_name, layer.weights)

                fh.write(output_text)

            count += 1

        all_values = individuals.values()

        total = 0
        count = 0

        for value in all_values:
            total += value
            count += 1

        print("Average:", total / count)


def simulate_2048(network):
    play_board = [[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]

    for _ in range(2):
        play_board = add_num(play_board)

    score = 0

    run = True

    while run:
        key = network.calculate(play_board)

        zero = False

        for row in play_board:
            if not all(row):
                zero = True
                break

        if not zero:
            over = True

            for i in range(3):
                for j in range(4):
                    if play_board[i][j] == play_board[i + 1][j]:
                        over = False
                        break

            for j in range(3):
                for i in range(4):
                    if play_board[i][j] == play_board[i][j + 1]:
                        over = False
                        break

            if over:
                run = False

        if key == "R":
            move_count = 0

            for i in range(4):
                row = play_board[i]

                frozen = False

                index = 3
                last_tile = 3

                while index >= 0:
                    if row[index] != 0:
                        if last_tile == 3:
                            tmp = row[index]
                            row[index] = 0
                            row[3] = tmp

                            if index != 3:
                                move_count += 1

                            last_tile -= 1
                        else:
                            if row[index] == row[last_tile + 1] and not frozen:
                                row[last_tile + 1] *= 2
                                score += row[last_tile + 1]

                                row[index] = 0
                                move_count += 1

                                frozen = True

                            else:
                                if row[index] == row[last_tile + 1]:
                                    frozen = False

                                tmp = row[index]
                                row[index] = 0
                                row[last_tile] = tmp

                                if index != last_tile:
                                    move_count += 1

                                last_tile -= 1

                    index -= 1

            if move_count:
                play_board = add_num(play_board)
                network.keystroke = 0
            else:
                network.keystroke += 1
        elif key == "L":
            move_count = 0

            for i in range(4):
                row = play_board[i]

                frozen = False

                index = 0
                last_tile = 0

                while index <= 3:
                    if row[index] != 0:
                        if last_tile == 0:
                            tmp = row[index]
                            row[index] = 0
                            row[0] = tmp

                            if index != 0:
                                move_count += 1

                            last_tile += 1
                        else:
                            if row[index] == row[last_tile - 1] and not frozen:
                                row[last_tile - 1] *= 2
                                score += row[last_tile - 1]

                                row[index] = 0
                                move_count += 1

                                frozen = True
                            else:
                                if row[index] == row[last_tile - 1]:
                                    frozen = False

                                tmp = row[index]
                                row[index] = 0
                                row[last_tile] = tmp

                                if index != last_tile:
                                    move_count += 1

                                last_tile += 1

                    index += 1

            if move_count:
                play_board = add_num(play_board)
                network.keystroke = 0
            else:
                network.keystroke += 1
        elif key == "D":
            move_count = 0

            for i in range(4):
                frozen = False

                index = 3
                last_tile = 3

                while index >= 0:
                    if play_board[index][i] != 0:
                        if last_tile == 3:
                            tmp = play_board[index][i]
                            play_board[index][i] = 0
                            play_board[3][i] = tmp

                            if index != 3:
                                move_count += 1

                            last_tile -= 1
                        else:
                            if play_board[index][i] == play_board[last_tile + 1][i] and not frozen:
                                play_board[last_tile + 1][i] *= 2
                                score += play_board[last_tile + 1][i]

                                play_board[index][i] = 0
                                move_count += 1

                                frozen = True
                            else:
                                if play_board[index][i] == play_board[last_tile + 1][i]:
                                    frozen = False

                                tmp = play_board[index][i]
                                play_board[index][i] = 0
                                play_board[last_tile][i] = tmp

                                if index != last_tile:
                                    move_count += 1

                                last_tile -= 1

                    index -= 1

            if move_count:
                play_board = add_num(play_board)
                network.keystroke = 0
            else:
                network.keystroke += 1
        elif key == "U":
            move_count = 0

            for i in range(4):
                frozen = False

                index = 0
                last_tile = 0

                while index <= 3:
                    if play_board[index][i] != 0:
                        if last_tile == 0:
                            tmp = play_board[index][i]
                            play_board[index][i] = 0
                            play_board[0][i] = tmp

                            if index != 0:
                                move_count += 1

                            last_tile += 1
                        else:
                            if play_board[index][i] == play_board[last_tile - 1][i] and not frozen:
                                play_board[last_tile - 1][i] *= 2
                                score += play_board[last_tile - 1][i]

                                play_board[index][i] = 0
                                move_count += 1

                                frozen = True
                            else:
                                if play_board[index][i] == play_board[last_tile - 1][i]:
                                    frozen = False

                                tmp = play_board[index][i]
                                play_board[index][i] = 0
                                play_board[last_tile][i] = tmp

                                if index != last_tile:
                                    move_count += 1

                                last_tile += 1

                    index += 1

            if move_count:
                play_board = add_num(play_board)
                network.keystroke = 0
            else:
                network.keystroke += 1

    return score


def main():
    NEAT()


if __name__ == "__main__":
    main()
