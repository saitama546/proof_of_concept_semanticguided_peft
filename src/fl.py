#The server holds the global model. The clients receive copies of the model. 
# Client 1 computes adapter gradients on private data.
from __future__ import annotations

import copy
from typing import List

import torch
import torch.nn.functional as F


class FLServer:
    """
    Description: Minimal FL server.
    INPUT: global model datatype: torch.nn.Module
     - For Step 1, we only test whether the server can send the global model to the client.
     - In real FL, the server would also aggregate gradients from clients and update the global model.  
     OUTPUT: a copy of the global model sent to the client. datatype: torch.nn.Module
    """

    def __init__(self, global_model: torch.nn.Module):
        """
        Description: Initialize the FL server with a global model.
        INPUT: global_model: a PyTorch model representing the global model on the server datatype: torch.nn.Module
         - For Step 1, we just store the global model on the server.
         - In real FL, the global model would be initialized and updated based on client updates.  
         - In this toy implementation, we just store the global model and send copies to clients.   
        OUTPUT: None
         - The server does not return anything upon initialization. 
        """
        self.global_model = global_model

    def send_model_to_client(self) -> torch.nn.Module:
        """
        Description: Send a copy of the global model to the client.
        INPUT: None
         - The server does not require any input to send the model to the client.
        OUTPUT: a copy of the global model sent to the client. datatype: torch.nn.Module
         - For Step 1, we just return a deep copy of the global model to simulate sending the model to the client.
         - In real FL, the server would serialize the model and send it over the network to the client. Here we just return a copy of the model in memory.  
         - The client would then receive this model and use it for local training.
        """
        return copy.deepcopy(self.global_model)


class FLClient:
    """
    Description: Minimal FL client.
    INPUT: client_id: int, model: torch.nn.Module, device: torch.device
     - For Step 1, we only test whether the client can receive the global model and compute gradients on private data.
     - In real FL, the client would also send gradients back to the server.  
     OUTPUT: a list of gradients with respect to adapter parameters. datatype: List[torch.Tensor]
    """

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        device: torch.device,
    ):
        """
        Description: Initialize the FL client with a client ID, a model, and a device.
        INPUT: client_id: an integer representing the unique ID of the client datatype: int
         - This is used to identify the client in a real FL setting. For Step 1, we just store the client ID for reference.
        model: a PyTorch model representing the local model on the client datatype: torch.nn.Module
         - For Step 1, we receive a copy of the global model from the server and store it as the local model on the client.
         - In real FL, the client would update this model based on local training and send updates back to the server. Here we just use it for local gradient computation.
        device: a PyTorch device (e.g., 'cpu' or 'cuda') representing where the client's computations will take place datatype: torch.device
         - This is used to specify where the client's computations will be performed. For Step 1, we just store the device for reference and move the model to the specified
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device

    def compute_adapter_gradient(
        self,
        private_x: torch.Tensor,
        private_y: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Description: Compute gradients with respect to adapter parameters on private data.
        INPUT: private_x: a tensor representing the input data on the client datatype: torch.Tensor
         - This is the input data that the client uses for local training. For Step 1, we just use this data to compute gradients. In real FL, this would be the client's private data that is not shared with the server or other clients.
        private_y: a tensor representing the labels corresponding to the input data on the client datatype: torch.Tensor
         - This is the labels corresponding to the input data that the client uses for local training. For Step 1, we just use this data to compute gradients. In real FL, this would be the client's private labels that are not shared with the server or other clients.
         - The client uses this data to compute the loss and gradients for local training. In real FL, the client would then send these gradients back to the server for aggregation and global model updates. Here we just return the gradients for testing purposes.  
            - The client would typically perform multiple local training steps and send the final gradients back to the server. Here we just compute and return the gradients for one step for testing purposes.
            - The client would also typically perform some local optimization (e.g., SGD) on the model parameters using these gradients before sending updates back to the server. Here we just return the raw gradients for testing purposes.
            - The client would also typically perform some local regularization or other techniques to improve training on private data. Here we just compute the gradients without any additional techniques for testing purposes.             
            - The client would also typically perform some local evaluation on a validation set to monitor training progress. Here we just compute the gradients without any evaluation for testing purposes.
            - The client would also typically perform some local data augmentation or other techniques to improve training on private data. Here we just compute the gradients without any additional techniques for testing purposes.

            OUTPUT: a list of gradients with respect to adapter parameters. datatype: List[torch.Tensor]    
        """
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(private_x)
        loss = F.cross_entropy(logits, private_y)

        adapter_params = self.model.get_adapter_parameters()

        adapter_grads = torch.autograd.grad(
            loss,
            adapter_params,
            create_graph=False,
            retain_graph=False,
        )

        return [g.detach().clone() for g in adapter_grads]