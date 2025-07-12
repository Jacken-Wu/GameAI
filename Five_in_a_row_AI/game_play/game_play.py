import torch
import logging


class GameBoard:
    PlayerAI = 0
    PlayerHuman = 1


    def __init__(self):
        self.board = torch.zeros(2, 15, 15)


    def reset(self):
        self.board = torch.zeros(2, 15, 15)


    def check_win(self, player: int) -> torch.Tensor:
        """
        Check if the given player has won the game.
        @param player: The player to check for win. 0 for player AI, 1 for player Human.
        @return: A board mask of the winning positions. Shape: (15, 15). Full of 1 when deadlock.
        """
        board = self.board.clone()
        if board.sum() == 225:
            logging.info("Game deadlock.")
            return torch.ones(15, 15)

        check_board = board[player]
        win_mark = torch.zeros(15, 15)
        for row in range(15):
            for col in range(11):
                # Check -
                if check_board[row][col] == 1 and check_board[row][col+1] == 1 and check_board[row][col+2] == 1 and check_board[row][col+3] == 1 and check_board[row][col+4] == 1:
                    win_mark[row][col:col+5] = 1
                # Check |, reverse row and col
                if check_board[col][row] == 1 and check_board[col+1][row] == 1 and check_board[col+2][row] == 1 and check_board[col+3][row] == 1 and check_board[col+4][row] == 1:
                    win_mark[col][row] = 1
                    win_mark[col+1][row] = 1
                    win_mark[col+2][row] = 1
                    win_mark[col+3][row] = 1
                    win_mark[col+4][row] = 1
                # Check \
                if row < 11:
                    if check_board[row][col] == 1 and check_board[row+1][col+1] == 1 and check_board[row+2][col+2] == 1 and check_board[row+3][col+3] == 1 and check_board[row+4][col+4] == 1:
                        win_mark[row][col] = 1
                        win_mark[row+1][col+1] = 1
                        win_mark[row+2][col+2] = 1
                        win_mark[row+3][col+3] = 1
                        win_mark[row+4][col+4] = 1
                # Check /
                if row > 3:
                    if check_board[row][col] == 1 and check_board[row-1][col+1] == 1 and check_board[row-2][col+2] == 1 and check_board[row-3][col+3] == 1 and check_board[row-4][col+4] == 1:
                        win_mark[row][col] = 1
                        win_mark[row-1][col+1] = 1
                        win_mark[row-2][col+2] = 1
                        win_mark[row-3][col+3] = 1
                        win_mark[row-4][col+4] = 1
        if torch.sum(win_mark) > 0:
            logging.debug(f"Win mark: {win_mark}")
        return win_mark
    
    
    def check_4_in_a_row(self, player: int, row: int, col: int, given_board: torch.Tensor = None) -> bool:
        """
        Check if the given player has 4 in a row at the given position.
        @param player: The player to check for 4 in a row. 0 for player AI, 1 for player Human.
        @return: True if the player has 4 in a row.
        """
        if given_board is None:
            check_board = self.board.clone()[player]
        else:
            check_board = given_board.clone()

        # Check -
        if col < 12 and check_board[row][col] == 1 and check_board[row][col+1] == 1 and check_board[row][col+2] == 1 and check_board[row][col+3] == 1:
            return True
        if 0 < col < 13 and check_board[row][col - 1] == 1 and check_board[row][col] == 1 and check_board[row][col + 1] == 1 and check_board[row][col + 2] == 1:
            return True
        if 1 < col < 14 and check_board[row][col - 2] == 1 and check_board[row][col - 1] == 1 and check_board[row][col] == 1 and check_board[row][col + 1] == 1:
            return True
        if col > 2 and check_board[row][col - 3] == 1 and check_board[row][col - 2] == 1 and check_board[row][col - 1] == 1 and check_board[row][col] == 1:
            return True

        # Check |, reverse row and col
        if col < 12 and check_board[col][row] == 1 and check_board[col+1][row] == 1 and check_board[col+2][row] == 1 and check_board[col+3][row] == 1:
            return True
        if 0 < col < 13 and check_board[col - 1][row] == 1 and check_board[col][row] == 1 and check_board[col + 1][row] == 1 and check_board[col + 2][row] == 1:
            return True
        if 1 < col < 14 and check_board[col - 2][row] == 1 and check_board[col - 1][row] == 1 and check_board[col][row] == 1 and check_board[col + 1][row] == 1:
            return True
        if col > 2 and check_board[col - 3][row] == 1 and check_board[col - 2][row] == 1 and check_board[col - 1][row] == 1 and check_board[col][row] == 1:
            return True

        # Check \
        if row < 12 and col < 12 and check_board[row][col] == 1 and check_board[row+1][col+1] == 1 and check_board[row+2][col+2] == 1 and check_board[row+3][col+3] == 1:
            return True
        if 0 < row < 13 and 0 < col < 13 and check_board[row - 1][col - 1] == 1 and check_board[row][col] == 1 and check_board[row + 1][col + 1] == 1 and check_board[row + 2][col + 2] == 1:
            return True
        if 1 < row < 14 and 1 < col < 14 and check_board[row - 2][col - 2] == 1 and check_board[row - 1][col - 1] == 1 and check_board[row][col] == 1 and check_board[row + 1][col + 1] == 1:
            return True
        if row > 2 and col > 2 and check_board[row - 3][col - 3] == 1 and check_board[row - 2][col - 2] == 1 and check_board[row - 1][col - 1] == 1 and check_board[row][col] == 1:
            return True

        # Check /
        if row > 2 and col < 12 and check_board[row][col] == 1 and check_board[row - 1][col + 1] == 1 and check_board[row - 2][col + 2] == 1 and check_board[row - 3][col + 3] == 1:
            return True
        if 1 < row < 14 and 0 < col < 13 and check_board[row + 1][col - 1] == 1 and check_board[row][col] == 1 and check_board[row - 1][col + 1] == 1 and check_board[row - 2][col + 2] == 1:
            return True
        if 0 < row < 13 and 1 < col < 14 and check_board[row + 2][col - 2] == 1 and check_board[row + 1][col - 1] == 1 and check_board[row][col] == 1 and check_board[row - 1][col + 1] == 1:
            return True
        if row < 12 and col > 2 and check_board[row + 3][col - 3] == 1 and check_board[row + 2][col - 2] == 1 and check_board[row + 1][col - 1] == 1 and check_board[row][col] == 1:
            return True

        return False
    
    
    def check_3_in_a_row(self, player: int, row: int, col: int) -> bool:
        """
        Check if the given player has 3 in a row at the given position.
        @param player: The player to check for 3 in a row. 0 for player AI, 1 for player Human.
        @return: True if the player has 3 in a row.
        """
        check_board = self.board.clone()[player]

        # Check -
        if col < 13 and check_board[row][col] == 1 and check_board[row][col + 1] == 1 and check_board[row][col + 2] == 1:
            return True
        if 0 < col < 14 and check_board[row][col - 1] == 1 and check_board[row][col] == 1 and check_board[row][col + 1] == 1:
            return True
        if col > 1 and check_board[row][col - 2] == 1 and check_board[row][col - 1] == 1 and check_board[row][col] == 1:
            return True

        # Check |, reverse row and col
        if col < 13 and check_board[col][row] == 1 and check_board[col + 1][row] == 1 and check_board[col + 2][row] == 1:
            return True
        if 0 < col < 14 and check_board[col - 1][row] == 1 and check_board[col][row] == 1 and check_board[col + 1][row] == 1:
            return True
        if col > 1 and check_board[col - 2][row] == 1 and check_board[col - 1][row] == 1 and check_board[col][row] == 1:
            return True

        # Check \
        if row < 13 and col < 13 and check_board[row][col] == 1 and check_board[row + 1][col + 1] == 1 and check_board[row + 2][col + 2] == 1:
            return True
        if 0 < row < 14 and 0 < col < 14 and check_board[row - 1][col - 1] == 1 and check_board[row][col] == 1 and check_board[row + 1][col + 1] == 1:
            return True
        if row > 1 and col > 1 and check_board[row - 2][col - 2] == 1 and check_board[row - 1][col - 1] == 1 and check_board[row][col] == 1:
            return True

        # Check /
        if row > 1 and col < 13 and check_board[row][col] == 1 and check_board[row - 1][col + 1] == 1 and check_board[row - 2][col + 2] == 1:
            return True
        if 0 < row < 14 and 0 < col < 14 and check_board[row + 1][col - 1] == 1 and check_board[row][col] == 1 and check_board[row - 1][col + 1] == 1:
            return True
        if row < 13 and col > 1 and check_board[row + 2][col - 2] == 1 and check_board[row + 1][col - 1] == 1 and check_board[row][col] == 1:
            return True
        
        return False
    
    
    def block_opponent_3(self, player: int, row: int, col: int) -> bool:
        """
        Block the opponent's move at the given position.
        @param player: The player making the move. 0 for player AI, 1 for player Human.
        @param row: The row of the move.
        @param col: The column of the move.
        @return: True if the move is blocked.
        """
        if self.board[player][row][col] == 0:
            return False

        check_board = self.board.clone()[1 - player]
        check_board[row][col] = 1
        
        if self.check_4_in_a_row(player=1-player, row=row, col=col, given_board=check_board):
            return True

        return False
    
    
    def block_opponent_4(self, player: int, row: int, col: int) -> bool:
        """
        Block the opponent's move at the given position.
        @param player: The player making the move. 0 for player AI, 1 for player Human.
        @param row: The row of the move.
        @param col: The column of the move.
        @return: True if the move is blocked.
        """
        if self.board[player][row][col] == 0:
            return False

        check_board = self.board.clone()[1 - player]
        check_board[row][col] = 1
        
        # Check -
        if col < 11 and check_board[row][col + 1] == 1 and check_board[row][col+2] == 1 and check_board[row][col + 3] == 1 and check_board[row][col + 4] == 1:
            return True
        if 0 < col < 12 and check_board[row][col - 1] == 1 and check_board[row][col + 1] == 1 and check_board[row][col + 2] == 1 and check_board[row][col + 3] == 1:
            return True
        if 1 < col < 13 and check_board[row][col - 2] == 1 and check_board[row][col - 1] == 1 and check_board[row][col + 1] == 1 and check_board[row][col + 2] == 1:
            return True
        if 2 < col < 14 and check_board[row][col - 3] == 1 and check_board[row][col - 2] == 1 and check_board[row][col - 1] == 1 and check_board[row][col + 1] == 1:
            return True
        if col > 3 and check_board[row][col - 4] == 1 and check_board[row][col - 3] == 1 and check_board[row][col - 2] == 1 and check_board[row][col - 1] == 1:
            return True

        # Check |, reverse row and col
        if col < 11 and check_board[col + 1][row] == 1 and check_board[col + 2][row] == 1 and check_board[col + 3][row] == 1 and check_board[col + 4][row] == 1:
            return True
        if 0 < col < 12 and check_board[col - 1][row] == 1 and check_board[col + 1][row] == 1 and check_board[col + 2][row] == 1 and check_board[col + 3][row] == 1:
            return True
        if 1 < col < 13 and check_board[col - 2][row] == 1 and check_board[col - 1][row] == 1 and check_board[col + 1][row] == 1 and check_board[col + 2][row] == 1:
            return True
        if 2 < col < 14 and check_board[col - 3][row] == 1 and check_board[col - 2][row] == 1 and check_board[col - 1][row] == 1 and check_board[col + 1][row] == 1:
            return True
        if col > 3 and check_board[col - 4][row] == 1 and check_board[col - 3][row] == 1 and check_board[col - 2][row] == 1 and check_board[col - 1][row] == 1:
            return True

        # Check \
        if row < 11 and col < 11 and check_board[row + 1][col + 1] == 1 and check_board[row + 2][col + 2] == 1 and check_board[row + 3][col + 3] == 1 and check_board[row + 4][col + 4] == 1:
            return True
        if 0 < row < 12 and 0 < col < 12 and check_board[row - 1][col - 1] == 1 and check_board[row + 1][col + 1] == 1 and check_board[row + 2][col + 2] == 1 and check_board[row + 3][col + 3] == 1:
            return True
        if 1 < row < 13 and 1 < col < 13 and check_board[row - 2][col - 2] == 1 and check_board[row - 1][col - 1] == 1 and check_board[row + 1][col + 1] == 1 and check_board[row + 2][col + 2] == 1:
            return True
        if 2 < row < 14 and 2 < col < 14 and check_board[row - 3][col - 3] == 1 and check_board[row - 2][col - 2] == 1 and check_board[row - 1][col - 1] == 1 and check_board[row + 1][col + 1] == 1:
            return True
        if row > 3 and col > 3 and check_board[row - 4][col - 4] == 1 and check_board[row - 3][col - 3] == 1 and check_board[row - 2][col - 2] == 1 and check_board[row - 1][col - 1] == 1:
            return True

        # Check /
        if row > 3 and col < 11 and check_board[row - 1][col + 1] == 1 and check_board[row - 2][col + 2] == 1 and check_board[row - 3][col + 3] == 1 and check_board[row - 4][col + 4] == 1:
            return True
        if 2 < row < 14 and 0 < col < 12 and check_board[row + 1][col - 1] == 1 and check_board[row - 1][col + 1] == 1 and check_board[row - 2][col + 2] == 1 and check_board[row - 3][col + 3] == 1:
            return True
        if 1 < row < 13 and 1 < col < 13 and check_board[row + 2][col - 2] == 1 and check_board[row + 1][col - 1] == 1 and check_board[row - 1][col + 1] == 1 and check_board[row - 2][col + 2] == 1:
            return True
        if 0 < row < 12 and 2 < col < 14 and check_board[row + 3][col - 3] == 1 and check_board[row + 2][col - 2] == 1 and check_board[row + 1][col - 1] == 1 and check_board[row - 1][col + 1] == 1:
            return True
        if row < 11 and col > 3 and check_board[row + 4][col - 4] == 1 and check_board[row + 3][col - 3] == 1 and check_board[row + 2][col - 2] == 1 and check_board[row + 1][col - 1] == 1:
            return True

        return False


    def move(self, player: int, row: int, col: int) -> bool:
        """
        Make a move on the board.
        @param player: The player making the move. 0 for player AI, 1 for player Human.
        @param row: The row to place the piece.
        @param col: The column to place the piece.
        @return: False if the move is invalid.
        """
        if self.board[0][row][col] == 1 or self.board[1][row][col] == 1:
            return False
        self.board[player][row][col] = 1
        return True
