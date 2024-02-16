from .gamerules.types.state import GameState


def display(state: GameState):
    board = state['board'].tolist()
    board_str = ''
    
    for row in reversed(board):
        for cell in row:
            match cell:
                case -1:
                    board_str += 'O'
                case 1:
                    board_str += 'X'
                case _:
                    board_str += '_'
            
        board_str += '\n'
    
    print(board_str)
