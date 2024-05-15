import threading

# 可以支持的最大 beam 参数上限，为了让生成的请求的group_req_id 和 sub_req_id 可以有直接的计算映射关系
# id 生成器，只会以 MAX_BEST_OF 的间隔生成id 作为 group_req_id, (sub_req_id // MAX_BEST_OF * MAX_BEST_OF) 即可
# 重新得到group_req_id

MAX_BEST_OF = 8


class ReqIDGenerator:
    def __init__(self):
        self.current_id = 0
        self.lock = threading.Lock()

    def generate_id(self):
        with self.lock:
            id = self.current_id
            self.current_id += MAX_BEST_OF
        return id


def convert_sub_id_to_group_id(sub_req_id):
    return (sub_req_id // MAX_BEST_OF) * MAX_BEST_OF
