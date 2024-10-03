import socket
import pickle
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FLServer:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.global_model = SimpleNN()

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen()
        print(f"[Server] Listening on {self.host}:{self.port}...")

        while True:
            client_socket, address = server_socket.accept()
            print(f"[Server] Client connected from {address}")

            try:
                # 1. Send the global model to the client
                print("[Server] Sending global model to client...")
                serialized_model = pickle.dumps(self.global_model.state_dict())
                client_socket.sendall(serialized_model)
                print("[Server] Global model sent to client.")

                # 2. Receive client's model
                print("[Server] Waiting to receive updated model from client...")
                received_data = client_socket.recv(409600)  # Adjust buffer size if needed
                client_model = pickle.loads(received_data)
                print("[Server] Model received from client.")

                # Log received model parameters
                print("[Server] Received model parameters:")
                print(client_model)

                # 3. Perform federated averaging (for simplicity, using the client's model as the global model)
                self.global_model.load_state_dict(client_model)

                # 4. Optionally, send the updated global model back to the client
                print("[Server] Sending updated global model back to client...")
                client_socket.sendall(pickle.dumps(self.global_model.state_dict()))
                print("[Server] Updated global model sent.")

            except Exception as e:
                print(f"[Server] Error: {e}")
            finally:
                client_socket.close()

if __name__ == "__main__":
    server = FLServer()
    server.start()
