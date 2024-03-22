import pygame
from random import randint
import tkinter as tk
from tkinter import messagebox
import math

banner_height = 100


def play_vars(win):
    if win.get_width() < win.get_height() - banner_height:
        play_x = 5
        play_size = win.get_width() - 10
        play_y = 100 + (win.get_height() - banner_height - play_size) / 2
    else:
        play_y = banner_height + 5
        play_size = win.get_height() - banner_height - 10
        play_x = (win.get_width() - play_size) / 2

    return play_x, play_y, play_size


def draw_background(win):
    win.fill((251, 249, 237))


def draw_banner(win, x, size, score, best_score, font, font2):
    pygame.draw.rect(win, (187, 174, 158), (x + 19 * size / 25, banner_height / 4, 6 * size / 25, banner_height / 2))
    best_score_title = font.render("BEST", True, (231, 217, 206))
    text_rect = best_score_title.get_rect()
    text_rect.center = (x + 22 * size / 25, 3 * banner_height / 8)
    win.blit(best_score_title, text_rect)
    best_score_text = font2.render(str(best_score), True, (255, 255, 255))
    text_rect = best_score_text.get_rect()
    text_rect.center = (x + 22 * size / 25, 5 * banner_height / 8)
    win.blit(best_score_text, text_rect)

    pygame.draw.rect(win, (187, 174, 158), (x + size / 2, banner_height / 4, 6 * size / 25, banner_height / 2))
    score_title = font.render("SCORE", True, (231, 217, 206))
    text_rect = score_title.get_rect()
    text_rect.center = (x + 31 * size / 50, 3 * banner_height / 8)
    win.blit(score_title, text_rect)
    score_text = font2.render(str(score), True, (255, 255, 255))
    text_rect = score_text.get_rect()
    text_rect.center = (x + 31 * size / 50, 5 * banner_height / 8)
    win.blit(score_text, text_rect)

    pygame.draw.rect(win, (187, 174, 158), (x + size / 12, banner_height / 4, size / 3, banner_height / 2))
    new_game = font.render("NEW GAME", True, (255, 255, 255))
    text_rect = new_game.get_rect()
    text_rect.center = (x + size / 4, banner_height / 2)
    win.blit(new_game, text_rect)


def draw_board(win, x, y, size, board, font):
    pygame.draw.rect(win, (189, 173, 157), (x, y, size, size))

    border_size = size / 25
    piece_size = size / 5

    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                color = (205, 192, 180)
            elif board[i][j] == 2:
                color = (238, 228, 218)
            elif board[i][j] == 4:
                color = (237, 223, 196)
            elif board[i][j] == 8:
                color = (244, 177, 122)
            elif board[i][j] == 16:
                color = (247, 150, 99)
            elif board[i][j] == 32:
                color = (245, 125, 98)
            elif board[i][j] == 64:
                color = (243, 95, 55)
            elif board[i][j] == 128:
                color = (236, 205, 114)
            elif board[i][j] == 256:
                color = (237, 202, 100)
            elif board[i][j] == 512:
                color = (237, 198, 81)
            elif board[i][j] == 1024:
                color = (237, 198, 68)
            elif board[i][j] == 2048:
                color = (236, 194, 48)            
            elif board[i][j] == 4096:
                color = (254, 61, 62)
            else:
                color = (255, 32, 32)

            x_corner = x + border_size * (j + 1) + piece_size * j
            y_corner = y + border_size * (i + 1) + piece_size * i

            pygame.draw.rect(win, color, (x_corner, y_corner, piece_size, piece_size))

            if board[i][j] == 2 or board[i][j] == 4:
                text = font.render(str(board[i][j]), True, (119, 110, 101))
                text_rect = text.get_rect()
                text_rect.center = (x_corner + piece_size / 2 + 1, y_corner + piece_size / 2 + 1)

                win.blit(text, text_rect)
            elif board[i][j] != 0:
                text = font.render(str(board[i][j]), True, (249, 249, 247))
                text_rect = text.get_rect()
                text_rect.center = (x_corner + piece_size / 2, y_corner + piece_size / 2)

                win.blit(text, text_rect)


def add_num(board):
    zeroes = False

    for i in range(4):
        if not all(board[i]):
            zeroes = True

    if zeroes:
        if randint(0, 9) == 9:
            num = 4
        else:
            num = 2

        found = False

        while not found:
            pos = randint(0, 15)

            if board[pos // 4][pos % 4] == 0:
                board[pos // 4][pos % 4] = num
                found = True

    return board


def draw_composite(win, x, y, size, board, font, font2, font3, score, best_score):
    draw_background(win)
    draw_banner(win, x, size, score, best_score, font2, font3)
    draw_board(win, x, y, size, board, font)


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
        
        try:
            return self.keys[indices[self.keystroke]]
        except IndexError:
            self.keystroke = 0
            return ""


def main():
    model = "Generation_210/3827.2_Number_7.txt"

    with open(model, "r") as fh:
        contents = fh.read()

    model_layers = []
    layers = contents.split("\n\n")

    layers.pop()

    for layer in layers:
        parts = layer.split("\n")

        activation_func = parts[0][parts[0].find(":") + 2:]
        temp_layer = Layer(getattr(Functions, activation_func))

        weights_str = parts[1][parts[1].find(":") + 3:-1]
        weights_arr = []

        while weights_str.find("[") != -1:
            weights_arr.append(weights_str[weights_str.find("[") + 1:weights_str.find("]")])
            weights_str = weights_str[weights_str.find("]") + 1:]

        layer_weights = []

        for weight_arr in weights_arr:
            layer_weights.append([])

            layer_weight = weight_arr.split(", ")

            for weight in layer_weight:
                layer_weights[len(layer_weights) - 1].append(float(weight))

        temp_layer.weights = layer_weights

        model_layers.append(temp_layer)

    model = Individual(model_layers)

    pygame.init()

    pygame.font.init()

    win = pygame.display.set_mode((500, 600), pygame.RESIZABLE)

    play_x, play_y, play_size = play_vars(win)
    play_board = [[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]
   
    num_font = pygame.font.SysFont("Calibri", int(play_size / 12.5), True)
    font2 = pygame.font.SysFont("Calibri", int(banner_height / 4), True)
    font3 = pygame.font.SysFont("Calibri", int(banner_height / 3), True)

    for _ in range(2):
        play_board = add_num(play_board)

    score = 0

    fh = open("best_score.txt", "r")
    best_score = int(fh.read())
    fh.close()

    draw_composite(win, play_x, play_y, play_size, play_board, num_font, font2, font3, score, best_score)

    pygame.display.update()

    pygame.display.set_caption("2048")
    
    run = True
    closed_out = True

    while run:
        pygame.time.delay(200)

        zero = False

        key = model.calculate(play_board)

        if closed_out:
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
                    blank = tk.Tk()
                    blank.withdraw()
                    messagebox.showinfo("Game Over", "Game Over!")
                    closed_out = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

                fh = open("best_score.txt", "w")
                fh.write(str(best_score))
                fh.close()
            elif event.type == pygame.VIDEORESIZE:
                win = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

                play_x, play_y, play_size = play_vars(win)

                draw_composite(win, play_x, play_y, play_size, play_board, num_font, font2, font3, score, best_score)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (play_x + play_size / 12 < event.pos[0] < play_x + 5 * play_size / 12 and
                        banner_height / 4 < event.pos[1] < 3 * banner_height / 4):
                    score = 0
                    play_board = [[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]

                    for _ in range(2):
                        play_board = add_num(play_board)

                    draw_banner(win, play_x, play_size, score, best_score, font2, font3)
                    draw_board(win, play_x, play_y, play_size, play_board, num_font)

                    closed_out = True

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
                                if score > best_score:
                                    best_score = score
                                draw_banner(win, play_x, play_size, score, best_score, font2, font3)

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
                draw_board(win, play_x, play_y, play_size, play_board, num_font)
                model.keystroke = 0
            else:
                model.keystroke += 1
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
                                if score > best_score:
                                    best_score = score
                                draw_banner(win, play_x, play_size, score, best_score, font2, font3)

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
                draw_board(win, play_x, play_y, play_size, play_board, num_font)
                model.keystroke = 0
            else:
                model.keystroke += 1
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
                                if score > best_score:
                                    best_score = score
                                draw_banner(win, play_x, play_size, score, best_score, font2, font3)

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
                draw_board(win, play_x, play_y, play_size, play_board, num_font)
                model.keystroke = 0
            else:
                model.keystroke += 1
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
                                if score > best_score:
                                    best_score = score
                                draw_banner(win, play_x, play_size, score, best_score, font2, font3)

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
                draw_board(win, play_x, play_y, play_size, play_board, num_font)
                model.keystroke = 0
            else:
                model.keystroke += 1

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
