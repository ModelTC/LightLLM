import socket



def alloc_can_use_network_port(num=3):
    port_list = []
    for port in range(10000, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))
            if result != 0:
                port_list.append(port)
            
            if len(port_list) == num:
                return port_list
    return None