import pickle
import numpy as np
import torch
import json
import time
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)

class GNNSceneEmbeddingNetwork_LearnedEdgeVector(nn.Module):
    def __init__(self, 
                 object_feature_dim=518, 
                 num_relations=26,  # Number of unique relations
                 node_embedding_dim=128, 
                 edge_embedding_dim=32,
                 scene_embedding_dim=32):
        """
        GNNSceneEmbeddingNetwork_LearnedEdgeVector processes a scene graph with node features and
        categorical edge features to generate a single scene embedding.

        Args:
        object_feature_dim (int): Dimensionality of input features for each object.
        num_relations (int): The total number of unique relation types.
        node_embedding_dim (int): Dimensionality of the intermediate node embeddings.
        edge_embedding_dim (int): Dimensionality of the learned edge embeddings.
        scene_embedding_dim (int): Dimensionality of the final scene embedding.
        """
        super(GNNSceneEmbeddingNetwork_LearnedEdgeVector, self).__init__()

        # 1. Node Feature Encoder: Maps high-dim raw features to a lower-dim space
        self.node_encoder = nn.Sequential(
            nn.Linear(object_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, node_embedding_dim)
        )
        
        # 2. Edge Feature Embedding: Creates a dense vector for each relation type
        # The layer learns a unique vector representation for each categorical relation.
        self.rel_embedding = nn.Embedding(num_relations, edge_embedding_dim)

        # 3. GNN Layers: Use GATConv which can incorporate edge features.
        # It updates each node's representation by attending to its neighbors' features
        # and their relations.
        self.conv1 = GATConv(in_channels=node_embedding_dim,
                             out_channels=node_embedding_dim,
                             edge_dim=edge_embedding_dim) # GATConv supports edge features directly
        
        self.conv2 = GATConv(in_channels=node_embedding_dim,
                             out_channels=node_embedding_dim,
                             edge_dim=edge_embedding_dim)

        # 4. Global Readout Network: Aggregates all node embeddings into a single vector
        # This is a key step to produce a fixed-size representation of the entire graph.
        self.readout_net = nn.Sequential(
            nn.Linear(node_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, scene_embedding_dim)
        )

    def forward(self, data):
        # Data object contains x, edge_index, and edge_attr
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. Encode node features
        x = self.node_encoder(x)  # Shape: (num_nodes, node_embedding_dim)

        # 2. Encode edge features using the embedding layer
        edge_attr = self.rel_embedding(edge_attr)  # Shape: (num_edges, edge_embedding_dim)

        # 3. Message Passing with GNN layers
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        
        # 4. Global pooling to get the scene embedding
        # PyG's global_mean_pool handles pooling over all nodes in the graph
        # This is the "readout" step that summarizes the entire graph.
        x = global_mean_pool(x, data.batch if data.batch is not None else torch.zeros(x.size(0), dtype=torch.long).to(device))

        # 5. Final scene-level encoding
        scene_embedding = self.readout_net(x)
        
        return scene_embedding

scene_file = "/home/docker_user/BeyondBareQueries/output/scenes/08.27.2025_18:36:01_edgeisaac_warehouse_objects.pkl"
from torch_geometric.data import Data

with open(scene_file, "rb") as f:
    data = pickle.load(f)

object_features = []
# 1. Process Node Features
for node in data['objects']:
    obj_id = node['node_id']
    clip_descriptor = np.array(node['clip_descriptor'])
    bbox_center = np.array(node['bbox_center'])
    bbox_extent = np.array(node['bbox_extent'])

    feature = np.concatenate([bbox_center, bbox_extent, clip_descriptor])
    object_features.append(feature)

# Convert node features to a PyTorch tensor
x_np = np.stack(object_features, axis=0)            # (N, 518)

x = torch.tensor(x_np, dtype=torch.float32).to(device)  # Shape: (num_objects, 518)

# 2. Process VL-SAT Edge Information
edge_index = []
edge_attr = []

for node in data['objects']:
    for edge in node['edges_vl_sat']:
        source_idx = edge['id_1']
        target_idx = edge['id_2']
        relation_str = edge['rel_name']
        relation_id = edge['rel_id']
        
        # Add edge connectivity
        edge_index.append([source_idx, target_idx])
        
        # Add numerical edge feature
        edge_attr.append(relation_id)

# Convert to PyTorch tensors
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device) # Shape: (2, num_edges)
edge_attr = torch.tensor(edge_attr, dtype=torch.long).to(device) # Shape: (num_edges)

# print dims
print(f"Node features shape: {x.shape}")
print(f"Edge index shape: {edge_index.shape}")
print(f"Edge attributes shape: {edge_attr.shape}")

# 3. Create a PyTorch Geometric Data object
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)


object_feature_dim = 518
num_relations = 26

# Instantiate and run one forward pass with random weights
model = GNNSceneEmbeddingNetwork_LearnedEdgeVector(
    object_feature_dim=object_feature_dim,
    num_relations=num_relations,
    node_embedding_dim=128,
    edge_embedding_dim=32,
    scene_embedding_dim=32
)

# IMPORTANT: Move the model to the same device as your data
model = model.to(device)

model.eval()

# Warm up GPU (important for accurate timing)
for _ in range(10):
    with torch.no_grad():
        _ = model(graph_data)

# Measure inference time
torch.cuda.synchronize()  # Ensure all GPU operations complete
start_time = time.time()

num_runs = 100
for _ in range(num_runs):
    with torch.no_grad():
        scene_emb = model(graph_data)

torch.cuda.synchronize()
end_time = time.time()

avg_inference_time = (end_time - start_time) / num_runs

print(f"Avg_100 inference time: {avg_inference_time} secs.")

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    scene_emb = model(graph_data)
peak_memory = torch.cuda.max_memory_allocated() / 1024**2

print(f"MAX GPU memory allocation: {peak_memory} MB")

with torch.no_grad():
    scene_emb = model(graph_data)
    
print("Scene embedding shape:", scene_emb.shape)
# print("Scene embedding (values):")
# print(scene_emb)