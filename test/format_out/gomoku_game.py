import os
import sys
import json
from pydantic import BaseModel, constr
from enum import Enum
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from format_out.impl import ChatSession
from format_out.impl import SamplingParams

# server model is Meta-Llama-3-8B-Instruct
MapSize = 9


class Piece(Enum):
    Black = "X"
    White = "O"
    Kong = "."


class Map(BaseModel):
    map_state: List[List[Piece]]

    def put_state(self, row, col, piece_state: Piece):
        if 1 <= row <= MapSize and 1 <= col <= MapSize:
            if self.map_state[row - 1][col - 1] is Piece.Kong:
                self.map_state[row - 1][col - 1] = piece_state
            else:
                raise Exception(f"location row {row} col {col} is not '{Piece.Kong.value}'")
        else:
            raise Exception(f"input row {row} col {col} is not ok, should 1 <= row <= {MapSize}, 1 <= col <= {MapSize}")

    def to_str(self):
        ans = " ".ljust(3)
        for i in range(MapSize):
            ans += str(i + 1).ljust(3)
        ans += "\n"
        for i in range(MapSize):
            ans += str(i + 1).ljust(3)
            for j in range(MapSize):
                state_c = self.map_state[i][j]
                ans += state_c.value.ljust(3)
            ans += "\n"
        return ans

    def check_victory(self, piece: Piece) -> bool:
        for row in range(len(self.map_state)):
            for col in range(len(self.map_state[row])):
                if self.map_state[row][col] == piece:
                    if (
                        self.check_direction(row, col, piece, 1, 0)
                        or self.check_direction(row, col, piece, 0, 1)
                        or self.check_direction(row, col, piece, 1, 1)
                        or self.check_direction(row, col, piece, 1, -1)
                    ):
                        return True
        return False

    def check_direction(self, row: int, col: int, piece: Piece, delta_row: int, delta_col: int) -> bool:
        count = 0
        for i in range(5):
            r = row + i * delta_row
            c = col + i * delta_col
            if 0 <= r < len(self.map_state) and 0 <= c < len(self.map_state[r]) and self.map_state[r][c] == piece:
                count += 1
            else:
                break
        return count == 5


class Loc(BaseModel):
    row: int
    col: int


Str_type = constr(min_length=3, max_length=200)


class ThoughtLoc(BaseModel):
    thoughts: List[Str_type]
    dest_loc: constr(min_length=3, max_length=200)
    row: int
    col: int


map = Map(map_state=[[Piece.Kong for _ in range(15)] for _ in range(15)])

system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a skilled Gomoku player
playing on a {MapSize} * {MapSize} board.
You hold the "X" pieces and play second,
while your opponent has "O" pieces.
The game character will provide feedback on the current status,
and you need to make decisions based on this information. The specific rules of the game are as follows:
Basic Rules
Pieces: The game uses black and white pieces, with black typically going first.
("O" represents black pieces, "X" represents white pieces)
Objective: Players need to place five of their same-colored pieces in a row,
either horizontally, vertically, or diagonally.
Placement: Players take turns placing their pieces in empty positions.
Victory Condition: The first player to form a continuous line of five same-colored pieces wins.
Draw: If the board is filled and no player has won, the game ends in a draw.
Game Flow
Opening: Black plays first, followed by alternating turns for white.
Placing Pieces: Each move must be made in an empty position and cannot overlap.
Check for Victory: After each move, check if any player has achieved five in a row.
<|eot_id|>"""
user_start = """<|start_header_id|>user<|end_header_id|>"""
user_end = """<|eot_id|>"""
assistant_start = """<|start_header_id|>assistant<|end_header_id|>"""
assistant_end = """<|eot_id|>"""
game_env_start = """<|start_header_id|>game_system<|end_header_id|>"""
game_env_end = """<|eot_id|>"""

chat_session = ChatSession(
    chat_his=system_prompt, url="http://localhost:8017/generate", sampling_param=SamplingParams(do_sample=True)
)
chat_session.sampling_param.top_p = 0.7
chat_session.sampling_param.top_k = 12
chat_session.disable_log = True
# 修改采样参数
chat_session.sampling_param.stop_sequences = [assistant_end, " " + assistant_end, "<|end_of_text|>", " <|end_of_text|>"]

current_is_user = True

for _ in range(MapSize * MapSize):
    chat_session.add_prompt(game_env_start)
    chat_session.add_prompt(map.to_str())
    print("current state:")
    print(map.to_str())
    has_win = False
    if map.check_victory(piece=Piece.Black):
        ans_str = f"\n'{Piece.Black.value}' has win\n restart new game"
        print(ans_str)
        chat_session.add_prompt(ans_str)
        has_win = True
    if map.check_victory(piece=Piece.White):
        ans_str = f"\n'{Piece.White.value}' has win\n restart new game"
        chat_session.add_prompt(ans_str)
        print(ans_str)
        has_win = True
    chat_session.add_prompt(game_env_end)

    if has_win is True:
        map.map_state = [[Piece.Kong for _ in range(15)] for _ in range(15)]
        continue

    if current_is_user:
        while True:
            try:
                cur_row = int(input("row:").strip())
                cur_col = int(input("col:").strip())
                cur_loc = Loc(row=cur_row, col=cur_col)
                map.put_state(cur_row, cur_col, piece_state=Piece.White)
                break
            except Exception as e:
                print(str(e))
                print("请重新输入")
                continue
        chat_session.add_prompt(user_start)
        chat_session.add_prompt(cur_loc.model_dump_json(indent=4))
        chat_session.add_prompt(user_end)
        current_is_user = False
    else:
        while True:
            chat_session.add_prompt(assistant_start)
            json_ans_str = chat_session.gen_json_object(ThoughtLoc, max_new_tokens=1000, prefix_regex=r"[\s]{0,20}")
            print("tmp:", json_ans_str)
            json_ans_str: str = json_ans_str.strip()
            json_ans_str = json_ans_str.replace("”", '"')  # 修复 json 格式问题
            json_ans = json.loads(json_ans_str)
            formatted_json = json.dumps(json_ans, indent=4, ensure_ascii=False)
            print("assistant:")
            print(formatted_json)
            chat_session.add_prompt(formatted_json)
            chat_session.add_prompt(assistant_end)

            try:
                cur_loc = ThoughtLoc(**json_ans)
                map.put_state(cur_loc.row, cur_loc.col, piece_state=Piece.Black)
                break
            except Exception as e:
                chat_session.add_prompt(game_env_start)
                chat_session.add_prompt(str(e))
                print("game env:")
                print(str(e))
                chat_session.add_prompt(game_env_end)
        current_is_user = True
