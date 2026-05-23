import random
import math

BOARD_SIZE = 5


def generate_position(size):
    x = random.randrange(0, size)
    y = random.randrange(0, size)
    return (x, y)

def calc_distance(pos1, pos2):
    diff_x = pos1[0] - pos2[0]
    diff_y = pos1[1] - pos2[1]
    return math.sqrt(diff_x**2 + diff_y**2)

def move_position(direction, pos):
    current_x, current_y = pos
    if direction == 'n':
        current_y -= 1
    elif direction == 's':
        current_y += 1
    elif direction == 'w':
        current_x -= 1
    elif direction == 'e':
        current_x += 1
    return (current_x, current_y)


def suikawari():
    suika_pos = generate_position(BOARD_SIZE)
    player_pos = generate_position(BOARD_SIZE)
    while suika_pos != player_pos:
        distance = calc_distance(suika_pos, player_pos)
        print("スイカへの距離:", distance)
        c = input("n:北に移動, s:南に移動, e:東に移動, w:西に移動")
        player_pos = move_position(c, player_pos)
    print("スイカを割りました！")

suikawari()