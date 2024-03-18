import socket
import threading
clients = []
def send_messages():
    global clients
    while True:
        response = input("Enter message to broadcast (Type 'bye' to exit): ")
        if response.lower().strip() == "bye":
            break
        for client_socket in clients:
            client_socket.send(response.encode())
def server():
    global clients
    host = socket.gethostname()
    port = 21042
    s = socket.socket()
    s.bind((host, port))
    s.listen(5) 
    print("Server listening for connections...")
    send_thread = threading.Thread(target=send_messages)
    send_thread.start()
    while True:
        client_socket, address = s.accept()
        clients.append(client_socket)
        print(f"Connected to: {address}")
server()
