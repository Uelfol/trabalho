import numpy as np
from os import system, name

ROWS = 6
COLUMNS = 7

WINDOW_SIZE = 4  # Tamanho da janela que vai se mover pelo tabuleiro
STRIDE = 1  # Passo com o qual a janela vai se mover

ALPHA = -np.Inf
BETA = np.Inf
STATES_EXPLORED = 0

# ----------------------------------------------------------------------------------
def clear():
    # para windows
    if name == 'nt':
        _ = system('cls')

    # para mac e linux(aqui, os.name eh 'posix')
    else:
        _ = system('clear')


# ----------------------------------------------------------------------------------
def create_board():
    board = np.zeros((ROWS, COLUMNS))
    return board


# ----------------------------------------------------------------------------------
def valid_location(board, column):
    return board[0][column] == 0  # Tem que verificar a linha 0 e não a última


# ----------------------------------------------------------------------------------
def drop_piece(board, column, piece):
    for r in range(ROWS - 1, -1, -1):  # Troquei a ordem da lista, para começar de baixo
        if board[r][column] == 0:
            board[r][column] = piece
            return


# ----------------------------------------------------------------------------------
def is_winning_move(board, piece):
    # verifica se existem quatro peças em linha na horizontal, vertical e diagonais
    for c in range(COLUMNS - 3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True
    for c in range(COLUMNS):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True
    for c in range(COLUMNS - 3):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True
    for c in range(COLUMNS - 3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True


def window_score(window, piece):  # Baseado no conceito das sliding windows das CNNs
    score = 0  # Vamos definir a pontuação de acordo com a quantidade de peças na janela
    adversary = 1  # Por padrão o adversario será o jogador

    conditions = [
        ((4, 0), 100),
        ((3, 1), 5),
        ((2, 2), 2),
        ((3, 1), - 4)
    ]

    if adversary == piece:
        adversary = 2  # Caso a peça seja do jogador, setamos o adversario para a IA (2)

    for r in range(WINDOW_SIZE):  # Verifica na horizontal a presença das peças
        for cond, cond_score in conditions:
            if np.count_nonzero(window[r, :] == piece) == cond[0] and np.count_nonzero(window[r, :] == 0) == cond[1]:
                score += cond_score

            if np.count_nonzero(window[r, :] == adversary) == cond[0] and \
                    np.count_nonzero(window[r, :] == 0) == cond[1]:
                score -= cond_score

    for c in range(WINDOW_SIZE):  # Verifica na horizontal a presença das peças
        for cond, cond_score in conditions:
            if np.count_nonzero(window[:, c] == piece) == cond[0] and np.count_nonzero(window[:, c] == 0) == cond[1]:
                score += cond_score

            if np.count_nonzero(window[:, c] == adversary) == cond[0] and \
                    np.count_nonzero(window[:, c] == 0) == cond[1]:
                score -= cond_score

    diagonals = [window.diagonal(), np.fliplr(window).diagonal()]

    for diagonal in diagonals:
        for cond, cond_score in conditions:
            if np.count_nonzero(diagonal == piece) == cond[0] and np.count_nonzero(diagonal == 0) == cond[1]:
                score += cond_score
            if np.count_nonzero(diagonal == adversary) == cond[0] and np.count_nonzero(diagonal == 0) == cond[1]:
                score -= cond_score

    print(f'Score: {score}')
    return score


def sliding_windows(board, piece):
    score = 0

    for r in range(ROWS - WINDOW_SIZE, -1, -STRIDE):
        for c in range(0, COLUMNS - WINDOW_SIZE + 1, STRIDE):
            window = board[r:r + WINDOW_SIZE, c:c + WINDOW_SIZE]
            score += window_score(window, piece)
            print('i')
    return score


# ----------------------------------------------------------------------------------
def minimax(board, depth, maximizing_player):
    if is_winning_move(board, 2):  # IA ganhou
        return (None, 100)
    elif is_winning_move(board, 1):  # jogador humano ganhou
        return (None, -100)
    elif len(get_valid_locations(board)) == 0:  # jogo empatado
        return (None, 0)
    elif depth == 0:  # profundidade máxima atingida
        return (None, sliding_windows(board, 2))

    valid_locations = get_valid_locations(board)
    if maximizing_player:
        value = -np.Inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            temp_board = board.copy()
            drop_piece(temp_board, col, 2)
            minimax_alpha_beta(temp_board, depth - 1, False, alpha, beta)[1]
            if new_score > value:
                value = new_score
                column = col
        return column, value

    else:  # minimizing player
        value = np.Inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            temp_board = board.copy()
            drop_piece(temp_board, col, 1)
            new_score = minimax_alpha_beta(temp_board, depth - 1, True, alpha, beta)[1]
            if new_score < value:
                value = new_score
                column = col
        return column, value
        
# ----------------------------------------------------------------------------------
def minimax_alpha_beta(board, depth, maximizing_player, alpha, beta):
    global STATES_EXPLORED
    global ALPHA
    global BETA

    if is_winning_move(board, 2):  # IA ganhou
        return (None, 100)
    elif is_winning_move(board, 1):  # jogador humano ganhou
        return (None, -100)
    elif len(get_valid_locations(board)) == 0:  # jogo empatado
        return (None, 0)

    # Heuristica
    elif depth == 0:  # profundidade máxima atingida
        return (None, 0)
        #return (None, evaluate(board))

    valid_locations = get_valid_locations(board)

    if maximizing_player:
        value = -np.Inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            temp_board = board.copy()
            drop_piece(temp_board, col, 2)

            STATES_EXPLORED += 1

            new_score = minimax_alpha_beta(temp_board, depth - 1, False,ALPHA,BETA)[1]
            if new_score > value:
                value = new_score
                column = col

            ###############################################
            ALPHA = max(ALPHA, value)
            if ALPHA >= BETA:
                print("Alpha: ", ALPHA, " Beta", BETA)
                break
            ###############################################

        return column, value

    else:  # minimizing player
        value = np.Inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            temp_board = board.copy()
            drop_piece(temp_board, col, 1)


            STATES_EXPLORED += 1

            new_score = minimax_alpha_beta(temp_board, depth - 1, True,ALPHA,BETA)[1]
            if new_score < value:
                value = new_score
                column = col

            ###############################################
            BETA = min(BETA, value)
            if ALPHA >= BETA:
                print("Alpha: ", ALPHA, " Beta", BETA)
                break
            ###############################################

        return column, value

# ----------------------------------------------------------------------------------
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMNS):
        if valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

# ----------------------------------------------------------------------------------
# CSI457 e CSI701
# Programa Principal
# Data: 06/05/2023
# ----------------------------------------------------------------------------------
board = create_board()
game_over = False
turn = 0

clear()
while not game_over:
    # Movimento do Jogador 1
    if turn == 0:
        col = int(input("Jogador 1, selecione a coluna (0-6):"))
        if valid_location(board, col):
            drop_piece(board, col, 1)
            if is_winning_move(board, 1):
                print("Jogador 1 Vence!! Parabéns!!")
                game_over = True

    # Movimento da IA
    else:
        # col, minimax_score = minimax(board, 4, True)  # A profundidade máxima da árvore é 4
        col, minimax_score = minimax_alpha_beta(board, 4, True, alpha, beta)
        if valid_location(board, col):
            drop_piece(board, col, 2)
            if is_winning_move(board, 2):
                print("Jogador 2 Vence!!!")
                game_over = True

    print(board)
    print(" ")
    turn += 1
    turn = turn % 2

print("Estados explorados:", STATES_EXPLORED)
