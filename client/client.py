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

class FLClient:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.model = SimpleNN()

    def start(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            print(f"[Client] Connecting to server at {self.host}:{self.port}...")
            client_socket.connect((self.host, self.port))

         
            print("[Client] Waiting to receive global model from server...")
            received_data = client_socket.recv(409600) 
            global_weights = pickle.loads(received_data)
            self.model.load_state_dict(global_weights)
            print("[Client] Global model received and loaded.")

           
            print("[Client] Initial model parameters:")
            print(self.model.state_dict())

        

            
            print("[Client] Sending updated model to the server...")
            client_socket.sendall(pickle.dumps(self.model.state_dict()))
            print("[Client] Updated model sent to the server.")

          
            print("[Client] Waiting to receive updated global model from server...")
            received_data = client_socket.recv(409600)  
            global_weights = pickle.loads(received_data)
            self.model.load_state_dict(global_weights)
            print("[Client] Updated global model received.")

            
            print("[Client] Updated model parameters:")
            print(self.model.state_dict())

        except Exception as e:
            print(f"[Client] Error: {e}")
        finally:
            client_socket.close()

if __name__ == "__main__":
    client = FLClient()
    client.start()
