import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import MLP, global_max_pool
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch

from .kinematics import ForwardKinematics


class SpatialBasicBlock(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='add', batch_norm=False, bias=True, **kwargs):
        super(SpatialBasicBlock, self).__init__(aggr=aggr, **kwargs)
        self.batch_norm = batch_norm
        # network architecture
        self.lin_f = nn.Linear(2*in_channels + edge_channels, out_channels, bias=bias)
        self.lin_s = nn.Linear(2*in_channels + edge_channels, out_channels, bias=bias)
        self.upsample = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out) if self.batch_norm else out
        out += self.upsample(x[1])
        return out

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))


class TemporalBasicBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dropout=0.5):
        super(TemporalBasicBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, 3),
                                           stride=1, padding=0, dilation=1))
        self.pad1 = torch.nn.ReplicationPad2d(((3-1) * 1, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, 3),
                                           stride=1, padding=0, dilation=2))
        self.pad2 = torch.nn.ReplicationPad2d(((3-1) * 2, 0, 0, 0))
        # self.net = nn.Sequential(self.pad1, self.conv1, self.relu, self.dropout,
        #                          self.pad2, self.conv2, self.relu, self.dropout)
        self.net = nn.Sequential(self.pad1, self.conv1, self.relu, self.dropout)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        return out + x


class Encoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Encoder, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=channels, out_channels=16, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=16, out_channels=32, edge_channels=dim)
        self.conv3 = TemporalBasicBlock(32, 32)
    
    def forward(self, data):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        out = self.conv1(data.x, data.edge_index, data.edge_attr)
        out = self.conv2(out, data.edge_index, data.edge_attr)
        # [T * joint_num, 32] ---> [T, joint_num, 32] ---> [joint_num, 32, T]
        out = out.reshape(data.num_graphs, -1, 32).permute(1, 2, 0)
        out = self.conv3(out).permute(2, 0, 1).reshape(-1, 32)
        return out


class Decoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Decoder, self).__init__()
        self.channels = channels
        self.conv1 = SpatialBasicBlock(in_channels=32, out_channels=16, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=16, out_channels=channels, edge_channels=dim)
        self.conv3 = TemporalBasicBlock(channels, channels)

    def forward(self, z, target):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        out = self.conv1(z, target.edge_index, target.edge_attr)
        out = self.conv2(out, target.edge_index, target.edge_attr)
        # [T * joint_num, channels] ---> [T, joint_num, channels] ---> [joint_num, channels, T]
        out = out.reshape(target.num_graphs, -1, self.channels).permute(1, 2, 0)
        out = self.conv3(out).permute(2, 0, 1).reshape(-1, self.channels)
        return out


class RootNet(torch.nn.Module):
    def __init__(self):
        super(RootNet, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=3, out_channels=16, edge_channels=3)
        self.conv2 = SpatialBasicBlock(in_channels=16+3, out_channels=3, edge_channels=3)

    def forward(self, data, target):
        out = self.conv1(data.x[:,6:], data.edge_index, data.edge_attr).view(data.num_graphs, -1, 16)
        out = torch.cat((out[:,:1,:].repeat(1,target.node_attr.shape[0]//target.num_graphs,1), target.node_attr.reshape(target.num_graphs,-1,3)), dim=-1).reshape(-1,16+3)
        out = self.conv2(out, target.edge_index, target.edge_attr).view(target.num_graphs, -1, 3)[:,0,:]
        return out


class MTNet(torch.nn.Module):
    def __init__(self, channel, dim, norm=True):
        super(MTNet, self).__init__()
        self.encoder = Encoder(channel, dim)
        self.decoder = Decoder(6, dim)
        self.global_nn = RootNet()
        self.fk = ForwardKinematics()
        self.norm = norm

    def forward(self, data, target):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        # encode source motion
        z = self.encoder(data)
        # decode target motion
        ang = self.decoder(z, target).view(target.num_graphs, -1, 6)
        if self.norm: ang = ang * target.ang_std.view(target.num_graphs, -1, 6) + target.ang_mean.view(target.num_graphs, -1, 6)
        # target pos
        pos = self.forward_kinematics(ang, target).view(target.num_graphs, -1, 3)
        root_pos = self.global_nn(data, target)
        if self.norm: root_pos = root_pos * target.root_std.view(target.num_graphs, 3) + target.root_mean.view(target.num_graphs, 3)
        pos += root_pos.unsqueeze(1)
        # fake target
        fake = target.clone()
        fake.x = torch.cat([ang, pos], dim=-1).view(-1, 9)
        if self.norm: fake.x = (fake.x - fake.mean) / fake.std
            
        return ang, pos, root_pos, None, fake

    def forward_kinematics(self, ang, target):
        pos = self.fk(ang, target.parent, target.offset, target.num_graphs).view(target.num_graphs, -1, 3)
        return pos


def split_data(data):
    data_list = data.to_data_list()
    l_hand_data = Batch.from_data_list([Data(
        x=data.l_hand_x if hasattr(data, 'l_hand_x') else None,
        edge_index=data.l_hand_edge_index,
        edge_attr=data.l_hand_edge_attr,
        parent=data.l_hand_parent,
        offset=data.l_hand_offset,
        mean=data.l_hand_mean if hasattr(data, 'l_hand_mean') else None,
        std=data.l_hand_std if hasattr(data, 'l_hand_std') else None,
        ang=data.l_hand_ang if hasattr(data, 'l_hand_ang') else None,
        ang_mean=data.l_hand_ang_mean if hasattr(data, 'l_hand_ang_mean') else None,
        ang_std=data.l_hand_ang_std if hasattr(data, 'l_hand_ang_std') else None,
        root_mean=data.l_hand_root_mean if hasattr(data, 'l_hand_root_mean') else None,
        root_std=data.l_hand_root_std if hasattr(data, 'l_hand_root_std') else None,
        node_attr=data.l_hand_node_attr,
        height=data.height,
    ) for data in data_list])
    r_hand_data = Batch.from_data_list([Data(
        x=data.r_hand_x if hasattr(data, 'r_hand_x') else None,
        edge_index=data.r_hand_edge_index,
        edge_attr=data.r_hand_edge_attr,
        parent=data.r_hand_parent,
        offset=data.r_hand_offset,
        mean=data.r_hand_mean if hasattr(data, 'r_hand_mean') else None,
        std=data.r_hand_std if hasattr(data, 'r_hand_std') else None,
        ang=data.r_hand_ang if hasattr(data, 'r_hand_ang') else None,
        ang_mean=data.r_hand_ang_mean if hasattr(data, 'r_hand_ang_mean') else None,
        ang_std=data.r_hand_ang_std if hasattr(data, 'r_hand_ang_std') else None,
        root_mean=data.r_hand_root_mean if hasattr(data, 'r_hand_root_mean') else None,
        root_std=data.r_hand_root_std if hasattr(data, 'r_hand_root_std') else None,
        node_attr=data.r_hand_node_attr,
        height=data.height,
    ) for data in data_list])
    return l_hand_data, r_hand_data


class UnifiedMTNet(torch.nn.Module):
    def __init__(self, channel, dim, norm=True):
        super(UnifiedMTNet, self).__init__()
        self.body_net = MTNet(channel, dim, norm)
        self.hand_net = MTNet(channel, dim, norm)

    def forward(self, data, target, return_global_pos=False, return_hand_trans=False):
        body_ang, body_pos, body_global_pos, body_trans, body_fake = self.body_net(data, target)
        if hasattr(data, 'l_hand_x') and hasattr(data, 'r_hand_x') and hasattr(target, 'l_hand_edge_index') and hasattr(target, 'r_hand_edge_index')\
            and data.l_hand_edge_index.shape[1] == target.l_hand_edge_index.shape[1] and data.r_hand_edge_index.shape[1] == target.r_hand_edge_index.shape[1]:
            l_hand_data, r_hand_data = split_data(data)
            l_hand_target, r_hand_target = split_data(target)
            l_hand_ang, _, _, l_hand_trans, _ = self.hand_net(l_hand_data, l_hand_target)
            r_hand_ang, _, _, r_hand_trans, _ = self.hand_net(r_hand_data, r_hand_target)
            hand_ang = torch.stack((l_hand_ang, r_hand_ang), dim=1)
        else:
            hand_ang, l_hand_trans, r_hand_trans = None, None, None
        if return_global_pos:
            return body_ang, body_pos, hand_ang, body_trans, body_global_pos
        if return_hand_trans:
            return body_ang, body_pos, hand_ang, body_trans, l_hand_trans, r_hand_trans, body_fake
        return body_ang, body_pos, hand_ang, body_trans, body_fake


class Discriminator(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Discriminator, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=channels, out_channels=16, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=16, out_channels=32, edge_channels=dim)
        self.conv3 = TemporalBasicBlock(32, 32)
        self.mlp = MLP([32, 16, 1], dropout=0.5, norm=None)

    def forward(self, data):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        out = self.conv1(data.x, data.edge_index, data.edge_attr)
        out = self.conv2(out, data.edge_index, data.edge_attr)
        # [T * joint_num, 32] ---> [T, joint_num, 32] ---> [joint_num, 32, T]
        out = out.reshape(data.num_graphs, -1, 32).permute(1, 2, 0)
        out = self.conv3(out).permute(2, 0, 1).reshape(-1, 32)
        out = global_max_pool(out, data.batch)
        out = self.mlp(out)
        out = torch.sigmoid(out)
        return out


class NodeEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(NodeEmbedding, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=in_channels, out_channels=out_channels//2, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=out_channels//2, out_channels=out_channels, edge_channels=dim)
    
    def forward(self, node_attr, edge_index, edge_attr):
        out = self.conv1(node_attr, edge_index, edge_attr)
        out = self.conv2(out, edge_index, edge_attr)
        return out


class GraphTransform(torch.nn.Module):
    def __init__(self):
        super(GraphTransform, self).__init__()
        self.emb_dim = 32
        self.node_embedding = NodeEmbedding(in_channels=3, out_channels=self.emb_dim, dim=3)

    def forward(self, x, data, target):
        # source node embedding
        src_edge_attr = data.edge_attr.view(data.num_graphs, -1, 3) / data.height.view(-1, 1, 1)
        src_edge_index, src_edge_attr = to_undirected(data.edge_index, src_edge_attr.view(-1, 3))
        src_node_emb = self.node_embedding(data.node_attr, src_edge_index, src_edge_attr)
        # target node embedding
        trg_edge_attr = target.edge_attr.view(target.num_graphs, -1, 3) / target.height.view(-1, 1, 1)
        trg_edge_index, trg_edge_attr = to_undirected(target.edge_index, trg_edge_attr.view(-1, 3))
        trg_node_emb = self.node_embedding(target.node_attr, trg_edge_index, trg_edge_attr)
        # reshape & concatenate
        src_node_emb = src_node_emb.view(data.num_graphs, -1, self.emb_dim)
        trg_node_emb = trg_node_emb.view(target.num_graphs, -1, self.emb_dim)
        # normalize to unit vector
        src_node_emb = src_node_emb / torch.norm(src_node_emb, dim=-1).unsqueeze(-1)
        trg_node_emb = trg_node_emb / torch.norm(trg_node_emb, dim=-1).unsqueeze(-1)
        # print(src_node_emb.shape, trg_node_emb.shape, src_node_emb[0], trg_node_emb[0])

        source_num_nodes = src_node_emb.shape[1]
        target_num_nodes = trg_node_emb.shape[1]
        transform_weights = torch.bmm(trg_node_emb, src_node_emb.permute(0, 2, 1))
        transform_weights = F.softmax(1e3*transform_weights, dim=-1)
        # print(transform_weights.shape, transform_weights[0])

        # transform
        out = x.view(data.num_graphs, source_num_nodes, -1)
        out = torch.bmm(transform_weights, out)

        # check shape
        assert out.shape[1] == target_num_nodes
        out = out.view(-1, out.shape[-1])

        return out, transform_weights


class TransformMTNet(torch.nn.Module):
    def __init__(self, channel, dim, norm=True):
        super(TransformMTNet, self).__init__()
        self.encoder = Encoder(channel, dim)
        self.transform = GraphTransform()
        self.decoder = Decoder(6, dim)
        self.global_nn = RootNet()
        self.fk = ForwardKinematics()
        self.norm = norm

    def forward(self, data, target):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index in spatial dimension [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        temporal_edge_index -- edge index in temporal dimension [2, num_edges]
        """
        # encode source motion
        z = self.encoder(data)
        transform_z, transform_weights = self.transform(z, data, target)
        # decode target motion
        ang = self.decoder(transform_z, target).view(target.num_graphs, -1, 6)
        if self.norm:
            ang = ang * target.ang_std.view(target.num_graphs, -1, 6) + target.ang_mean.view(target.num_graphs, -1, 6)
        else:
            ang = torch.bmm(transform_weights, data.ang.view(data.num_graphs, -1, 6))
            select_idx = torch.argmax(transform_weights[0], dim=0)
            mask_idx = [i for i in range(transform_weights.shape[1]) if (select_idx != i).all()]
            ang[:, mask_idx, :] = torch.tensor([1., 0., 0., 0., 1., 0.]).to(ang.device)
        # target pos
        pos = self.forward_kinematics(ang, target).view(target.num_graphs, -1, 3)
        # root_pos = global_max_pool(self.global_nn(transform_z, target), target.batch)
        # root_pos = data.x.reshape(data.num_graphs, -1, 9)[:, 0, 6:]
        root_pos = self.global_nn(data, target)
        if self.norm: root_pos = root_pos * target.root_std.view(target.num_graphs, 3) + target.root_mean.view(target.num_graphs, 3)
        pos += root_pos.unsqueeze(1)
        # fake target
        fake = target.clone()
        fake.x = torch.cat([ang, pos], dim=-1).view(-1, 9)
        if self.norm: fake.x = (fake.x - fake.mean) / fake.std
        return ang, pos, root_pos, transform_weights, fake

    def forward_kinematics(self, ang, target):
        pos = self.fk(ang, target.parent, target.offset, target.num_graphs).view(target.num_graphs, -1, 3)
        return pos


class UnifiedNet(torch.nn.Module):
    def __init__(self, channel, dim, norm=True):
        super(UnifiedNet, self).__init__()
        self.body_net = TransformMTNet(channel, dim, norm)
        self.hand_net = TransformMTNet(channel, dim, norm)

    def forward(self, data, target, return_global_pos=False, return_hand_trans=False):
        body_ang, body_pos, body_global_pos, body_trans, body_fake = self.body_net(data, target)
        if hasattr(data, 'l_hand_x') and hasattr(data, 'r_hand_x') and hasattr(target, 'l_hand_edge_index') and hasattr(target, 'r_hand_edge_index'):
            l_hand_data, r_hand_data = split_data(data)
            l_hand_target, r_hand_target = split_data(target)
            l_hand_ang, _, _, l_hand_trans, _ = self.hand_net(l_hand_data, l_hand_target)
            r_hand_ang, _, _, r_hand_trans, _ = self.hand_net(r_hand_data, r_hand_target)
            hand_ang = torch.stack((l_hand_ang, r_hand_ang), dim=1)
        else:
            hand_ang, l_hand_trans, r_hand_trans = None, None, None
        if return_global_pos:
            return body_ang, body_pos, hand_ang, body_trans, body_global_pos
        if return_hand_trans:
            return body_ang, body_pos, hand_ang, body_trans, l_hand_trans, r_hand_trans, body_fake
        return body_ang, body_pos, hand_ang, body_trans, body_fake