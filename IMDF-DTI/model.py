"""IMDF-DTI model extensions: X-Mask, DFGU, and RESCAL-Hybrid.

Sequence/graph backbone adapted from MIF-DTI. See THIRD_PARTY_NOTICES.md.
Module attribute names are retained for state-dict checkpoint compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from layers import *
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool
                                )
from config import hyperparameter

class GraphConvBlock(nn.Module):
    """
    Graph attention, normalization, and pooling block.
    """
    def __init__(self, in_channels=200, out_channels=200, num_heads=4, dropout=0.3):
        super(GraphConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout


        self.conv = GATConv(self.in_channels, self.out_channels//self.num_heads, self.num_heads, dropout=self.dropout)

        self.norm = LayerNorm(self.in_channels)

        self.readout = SAGPooling(self.out_channels, min_score=-1)

    def forward(self, x, edge_index, batch, edge_attr=None):

        x = F.elu(self.norm(x, batch))

        x = self.conv(x, edge_index, edge_attr)

        x, _, _, x_batch, _, _ = self.readout(x, edge_index, edge_attr=edge_attr, batch=batch)
        global_graph_emb = global_add_pool(x, x_batch)
        return x, global_graph_emb


class GraphInteractionBlock(nn.Module):
    """
    Graph-view drug/target interaction block adapted from the baseline.
    """
    def __init__(self, in_channels=200, out_channels=200, num_heads=5, dropout=0.4):
        super(GraphInteractionBlock, self).__init__()

        self.hidden_channels = out_channels // (num_heads*2)

        self.drug_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.1)

        self.prot_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.3)

        self.inter_conv = GATConv((in_channels, in_channels), self.hidden_channels, num_heads, dropout=dropout)

        self.drug_norm = LayerNorm(out_channels)
        self.prot_norm = LayerNorm(out_channels)

        self.drug_pool = GATConv(out_channels, out_channels//num_heads, num_heads)
        self.prot_pool = SAGPooling(out_channels, min_score=-1)


    def forward(self, atom_x, atom_edge_index, bond_x, atom_batch, \
                aa_x, aa_edge_index, aa_edge_attr, aa_batch, m2p_edge_index):


        atom_x_res = atom_x
        aa_x_res = aa_x


        atom_intra_x = self.drug_conv(atom_x, atom_edge_index, bond_x)

        atom_inter_x = self.inter_conv((aa_x, atom_x), m2p_edge_index[[1,0]])

        atom_x_tmp = torch.cat([atom_intra_x, atom_inter_x], -1)

        atom_x = F.elu(self.drug_norm(atom_x_tmp, atom_batch))


        aa_intra_x = self.prot_conv(aa_x, aa_edge_index, aa_edge_attr)

        aa_inter_x = self.inter_conv((atom_x, aa_x), m2p_edge_index)

        aa_x_tmp = torch.cat([aa_intra_x, aa_inter_x], -1)

        aa_x = F.elu(self.prot_norm(aa_x_tmp, aa_batch))


        atom_x = self.drug_pool(atom_x, atom_edge_index, bond_x)
        aa_x, _, _, aa_batch, _, _ = self.prot_pool(aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)


        atom_x = F.dropout(atom_x_res+F.elu(atom_x), 0.1, self.training)
        aa_x = F.dropout(aa_x_res+F.elu(aa_x), 0.1, self.training)


        drug_global_repr = global_add_pool(atom_x, atom_batch)
        prot_global_repr = global_add_pool(aa_x, aa_batch)

        return atom_x, aa_x, drug_global_repr, prot_global_repr

class SequenceInteractionBlock(nn.Module):
    """
    Sequence-view convolution and bidirectional cross-attention block.
    """
    def __init__(self, input_dim=200, conv=50, drug_kernel=[4, 6, 8], prot_kernel=[4, 8, 12]):
        super(SequenceInteractionBlock, self).__init__()
        self.attention_dim = conv * 4
        self.mix_attention_head = 5


        self.Drug_CNNs = get_CNNs(input_dim, conv, drug_kernel)
        self.Protein_CNNs = get_CNNs(input_dim, conv, prot_kernel)


        self.mix_attention_layer = nn.MultiheadAttention(self.attention_dim, self.mix_attention_head, batch_first=True, dropout=0.3)

    def forward(self, drugembed, proteinembed):


        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)


        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)


        drugConv = drugConv.permute(0, 2, 1)
        proteinConv = proteinConv.permute(0, 2, 1)


        drug_att, _ = self.mix_attention_layer(drugConv, proteinConv, proteinConv)
        protein_att, _ = self.mix_attention_layer(proteinConv, drugConv, drugConv)


        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5


        drugPool, _ = torch.max(drugConv, dim=1)
        proteinPool, _ = torch.max(proteinConv, dim=1)

        return drugConv, proteinConv, drugPool, proteinPool


class IMDFDTI(nn.Module):
    """
    IMDF-DTI: masked sequence/graph encoding and mask-aware interaction decoding.
    """
    def __init__(self, depth=3, device='cuda:0', hp=None):
        super(IMDFDTI, self).__init__()

        self.drug_in_channels = 43
        self.prot_in_channels = 33
        self.prot_evo_in_channels = 1280
        self.hidden_channels = 200
        self.depth = depth
        self.device = device


        if hp is None:
            hp = hyperparameter()

        self.hp = hp
        self.use_xmask = getattr(hp, "use_xmask", True)
        self.use_dfgu = getattr(hp, "use_dfgu", True)
        self.use_rescal_hybrid = getattr(hp, "use_rescal_hybrid", True)


        self.atom_type_encoder = Embedding(20, self.hidden_channels)

        self.atom_feat_encoder = MLP([self.drug_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True)

        self.bond_encoder = Embedding(10, self.hidden_channels)


        self.prot_evo = MLP([self.prot_evo_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True)

        self.prot_aa = MLP([self.prot_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True)


        self.blocks = nn.ModuleList([GraphInteractionBlock() for _ in range(depth)])


        self.drug_seq_emb = nn.Embedding(65, self.hidden_channels, padding_idx=0)
        self.prot_seq_emb = nn.Embedding(26, self.hidden_channels, padding_idx=0)

        self.blocks_1D = nn.ModuleList([SequenceInteractionBlock() for _ in range(depth)])


        if self.use_rescal_hybrid:
            self.attn = RESCAL_Hybrid(self.hidden_channels, self.depth * 2, alpha=0.5)
        else:
            self.attn = RESCAL(self.hidden_channels, self.depth * 2)


        self.mask_2d_drug = FeatureMask(self.hidden_channels)
        self.mask_2d_prot = FeatureMask(self.hidden_channels)

        self.mask_1d_drug = FeatureMask(self.hidden_channels)
        self.mask_1d_prot = FeatureMask(self.hidden_channels)


        self.dfgu_2d = DFGU()
        self.dfgu_1d = DFGU()


        self.to(device)

    def _fixed_weights(self, dfgu_module):
        """w/o DFGU 时使用固定 0.5/0.5 权重，并保持 latest_weights 存在，避免训练日志报错。"""
        w = torch.tensor([0.5, 0.5], device=self.device)
        dfgu_module.latest_weights = w
        return w


    def forward(self,data):


        atom_x, atom_x_feat, smiles_x, atom_edge_index, bond_x, mol_node_levels = \
            data.mol_x, data.mol_x_feat, data.mol_smiles_x, data.mol_edge_index, data.mol_edge_attr, data.mol_node_levels

        aa_x, aa_evo_x, seq_x, aa_edge_index, aa_edge_weight = \
            data.prot_node_aa, data.prot_node_evo, data.prot_seq_x, data.prot_edge_index, data.prot_edge_weight, \

        atom_batch, aa_batch = data.mol_x_batch, data.prot_node_aa_batch

        m2p_edge_index = data.m2p_edge_index


        atom_x = self.atom_type_encoder(atom_x.squeeze()) + self.atom_feat_encoder(atom_x_feat)
        bond_x = self.bond_encoder(bond_x)


        aa_x = self.prot_aa(aa_x) + self.prot_evo(aa_evo_x)
        aa_edge_attr = rbf(aa_edge_weight, D_max=1.0, D_count=self.hidden_channels, device=self.device)


        if self.use_xmask:
            atom_x, m_2d_d = self.mask_2d_drug(atom_x)
            aa_x, m_2d_p = self.mask_2d_prot(aa_x)
        else:
            m_2d_d = torch.ones(self.hidden_channels, device=self.device)
            m_2d_p = torch.ones(self.hidden_channels, device=self.device)


        if self.use_dfgu:
            w_2d = self.dfgu_2d(m_2d_d, m_2d_p)
        else:
            w_2d = self._fixed_weights(self.dfgu_2d)


        drug_repr = []
        prot_repr = []

        for i in range(self.depth):
            out = self.blocks[i](atom_x, atom_edge_index, bond_x, atom_batch, \
                                 aa_x, aa_edge_index, aa_edge_attr, aa_batch, \
                                 m2p_edge_index)
            atom_x, aa_x, drug_global_repr, prot_global_repr = out
            drug_global_repr = atom_x[mol_node_levels==2]


            drug_repr.append(atom_x[mol_node_levels==2] * w_2d[0])
            prot_repr.append(prot_global_repr * w_2d[1])


        atom_x_seq = self.drug_seq_emb(smiles_x)
        aa_x_seq = self.prot_seq_emb(seq_x)


        if self.use_xmask:
            atom_x_seq, m_1d_d = self.mask_1d_drug(atom_x_seq)
            aa_x_seq, m_1d_p = self.mask_1d_prot(aa_x_seq)
        else:
            m_1d_d = torch.ones(self.hidden_channels, device=self.device)
            m_1d_p = torch.ones(self.hidden_channels, device=self.device)


        if self.use_dfgu:
            w_1d = self.dfgu_1d(m_1d_d, m_1d_p)
        else:
            w_1d = self._fixed_weights(self.dfgu_1d)

        for i in range(self.depth):
            out_seq = self.blocks_1D[i](atom_x_seq, aa_x_seq)
            atom_x_seq, aa_x_seq, drug_seq_pool, prot_seq_pool = out_seq
            drug_repr.append(drug_seq_pool * w_1d[0])
            prot_repr.append(prot_seq_pool * w_1d[1])


        drug_repr = torch.stack(drug_repr, dim=-2)
        prot_repr = torch.stack(prot_repr, dim=-2)


        if self.use_rescal_hybrid:
            s_drug_avg = (w_2d[0] + w_1d[0]) / 2
            s_prot_avg = (w_2d[1] + w_1d[1]) / 2
            scores = self.attn(drug_repr, prot_repr, s_drug=s_drug_avg, s_prot=s_prot_avg)
        else:
            scores = self.attn(drug_repr, prot_repr)


        return scores
    def get_explainability_weights(self):
        """
        Return learned feature-mask scores, not atom/residue attributions.
        """
        with torch.no_grad():
            weights = {
                "mask_2d_drug": torch.sigmoid(self.mask_2d_drug.delta).cpu().numpy(),
                "mask_2d_prot": torch.sigmoid(self.mask_2d_prot.delta).cpu().numpy(),
                "mask_1d_drug": torch.sigmoid(self.mask_1d_drug.delta).cpu().numpy(),
                "mask_1d_prot": torch.sigmoid(self.mask_1d_prot.delta).cpu().numpy()
            }
        return weights

def get_m2p_edge_from_batch(atom_batch, aa_batch, node_level=None):
    """
    从Batch中获取药物-蛋白质交互边索引
    """

    mask = atom_batch.unsqueeze(1) == aa_batch.unsqueeze(0)
    if node_level is not None:
        mask = mask * (node_level==1).unsqueeze(1)
    a_idx, b_idx = torch.nonzero(mask, as_tuple=True)
    edge_list = torch.stack([a_idx, b_idx], dim=0)
    return edge_list


# Compatibility names for existing local scripts and previously serialized models.
MIFDTI = IMDFDTI
MIF_conv_block = GraphConvBlock
MIFBlock = GraphInteractionBlock
MIFBlock_1D = SequenceInteractionBlock
