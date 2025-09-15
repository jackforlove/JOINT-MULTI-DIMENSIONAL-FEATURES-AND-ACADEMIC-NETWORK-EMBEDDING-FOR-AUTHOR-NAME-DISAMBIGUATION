import  torch
from    torch import nn
from    torch.nn import functional as F
import numpy as np

class ASGC(nn.Module):


    def __init__(self, input_dim, output_dim):
        super(ASGC, self).__init__()

        self.layer_1 = nn.Linear(input_dim, output_dim)
        self.layer_2 = nn.Linear(output_dim, output_dim)
        self.layer_3= nn.Linear(output_dim, output_dim)
        self.layers_agcn = nn.Linear(output_dim,1)

    def scale(self, z):

        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled
    def forward(self, inputs):
        # inputs = F.normalize(inputs)

        h_1 = self.layer_1(inputs)
        h_2 = self.layer_2(h_1)
        h_3 = self.layer_3(h_2)

        H = torch.stack((h_1, h_2, h_3), 1)
        A = self.layers_agcn(H)
        A = F.softmax(A,dim=1)
        H = torch.permute(H,(0,2,1))
        out = torch.bmm(H,A).squeeze()
        # out = F.sigmoid(out)
        # out = self.scale(out)
        # out = F.normalize(out)

        # out = F.softmax(out)
        return out

class GAT(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_heads: int, dropout: float):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads,
                                            is_concat=True, dropout=dropout)
        self.layer2 = GraphAttentionLayer(n_hidden, n_hidden, n_heads,
                                            is_concat=True, dropout=dropout)
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        self.dcs = SampleDecoder(act=lambda x: x)
        # Final graph attention layer where we average the heads
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.layer2(x, adj_mat)
        # x = self.activation(x)
        out = F.sigmoid(x)
        # out = self.scale(out)
        out = F.normalize(out)



        # x = self.dropout(x)
        return out
class SampleDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid):
        super(SampleDecoder, self).__init__()
        self.act = act

    def forward(self, zx, zy):
        sim = (zx * zy).sum(1)
        sim = self.act(sim)
        # zx = F.normalize(zx)  # F.normalize只能处理两维的数据，L2归一化
        # zy = F.normalize(zy)
        # sim = zx.mm(zy.t())
        # sim = self.act(sim)

        return sim

class GetWeigit(nn.Module):
    def __init__(self, act=torch.sigmoid):
        super(GetWeigit, self).__init__()
        self.act = act
    def forward(self,zx,zy):
        dot = np.dot(zx,zy)
        zx_sum = np.sqrt(np.dot(zx,zx))
        zy_sum = np.sqrt(np.dot(zy,zy))
        cos = dot/(zx_sum*zy_sum)
        sim = self.act(cos)
        return sim

class GraphAttentionLayer(nn.Module):
    """
    ## Graph attention layer

    This is a single graph attention layer.
    A GAT is made up of multiple such layers.

    It takes
    $$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$,
    where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input
    and outputs
    $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$,
    where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.

        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # The initial transformation,
        # $$\overrightarrow{g^k_i} = \mathbf{W}^k \overrightarrow{h_i}$$
        # for each head.
        # We do single linear transformation and then split it up for each head.
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)

        # #### Calculate attention score
        #
        # We calculate these for each head $k$. *We have omitted $\cdot^k$ for simplicity*.
        #
        # $$e_{ij} = a(\mathbf{W} \overrightarrow{h_i}, \mathbf{W} \overrightarrow{h_j}) =
        # a(\overrightarrow{g_i}, \overrightarrow{g_j})$$
        #
        # $e_{ij}$ is the attention score (importance) from node $j$ to node $i$.
        # We calculate this for each head.
        #
        # $a$ is the attention mechanism, that calculates the attention score.
        # The paper concatenates
        # $\overrightarrow{g_i}$, $\overrightarrow{g_j}$
        # and does a linear transformation with a weight vector $\mathbf{a} \in \mathbb{R}^{2 F'}$
        # followed by a $\text{LeakyReLU}$.
        #
        # $$e_{ij} = \text{LeakyReLU} \Big(
        # \mathbf{a}^\top \Big[
        # \overrightarrow{g_i} \Vert \overrightarrow{g_j}
        # \Big] \Big)$$

        # First we calculate
        # $\Big[\overrightarrow{g_i} \Vert \overrightarrow{g_j} \Big]$
        # for all pairs of $i, j$.
        #
        # `g_repeat` gets
        # $$\{\overrightarrow{g_1}, \overrightarrow{g_2}, \dots, \overrightarrow{g_N},
        # \overrightarrow{g_1}, \overrightarrow{g_2}, \dots, \overrightarrow{g_N}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_repeat = g.repeat(n_nodes, 1, 1)
        # `g_repeat_interleave` gets
        # $$\{\overrightarrow{g_1}, \overrightarrow{g_1}, \dots, \overrightarrow{g_1},
        # \overrightarrow{g_2}, \overrightarrow{g_2}, \dots, \overrightarrow{g_2}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        # Now we concatenate to get
        # $$\{\overrightarrow{g_1} \Vert \overrightarrow{g_1},
        # \overrightarrow{g_1} \Vert \overrightarrow{g_2},
        # \dots, \overrightarrow{g_1}  \Vert \overrightarrow{g_N},
        # \overrightarrow{g_2} \Vert \overrightarrow{g_1},
        # \overrightarrow{g_2} \Vert \overrightarrow{g_2},
        # \dots, \overrightarrow{g_2}  \Vert \overrightarrow{g_N}, ...\}$$
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        # Reshape so that `g_concat[i, j]` is $\overrightarrow{g_i} \Vert \overrightarrow{g_j}$
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)

        # Calculate
        # $$e_{ij} = \text{LeakyReLU} \Big(
        # \mathbf{a}^\top \Big[
        # \overrightarrow{g_i} \Vert \overrightarrow{g_j}
        # \Big] \Big)$$
        # `e` is of shape `[n_nodes, n_nodes, n_heads, 1]`
        e = self.activation(self.attn(g_concat))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # We then normalize attention scores (or coefficients)
        # $$\alpha_{ij} = \text{softmax}_j(e_{ij}) =
        # \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$
        #
        # where $\mathcal{N}_i$ is the set of nodes connected to $i$.
        #
        # We do this by setting unconnected $e_{ij}$ to $- \infty$ which
        # makes $\exp(e_{ij}) \sim 0$ for unconnected pairs.
        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        # $$\overrightarrow{h'^k_i} = \sum_{j \in \mathcal{N}_i} \alpha^k_{ij} \overrightarrow{g^k_j}$$
        #
        # *Note:* The paper includes the final activation $\sigma$ in $\overrightarrow{h_i}$
        # We have omitted this from the Graph Attention Layer implementation
        # and use it on the GAT model to match with how other PyTorch modules are defined -
        # activation as a separate layer.
        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            # $$\overrightarrow{h'_i} = \Bigg\Vert_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            # $$\overrightarrow{h'_i} = \frac{1}{K} \sum_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.mean(dim=1)


class GraphAttentionV2Layer(nn.Module):
    """
    ## Graph attention v2 layer
    This is a single graph attention v2 layer.
    A GATv2 is made up of multiple such layers.
    It takes
    $$\mathbf{h} = \{ \overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N} \}$$,
    where $\overrightarrow{h_i} \in \mathbb{R}^F$ as input
    and outputs
    $$\mathbf{h'} = \{ \overrightarrow{h'_1}, \overrightarrow{h'_2}, \dots, \overrightarrow{h'_N} \}$$,
    where $\overrightarrow{h'_i} \in \mathbb{R}^{F'}$.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        * `share_weights` if set to `True`, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # The initial transformations,
        # $$\overrightarrow{{g_l}^k_i} = \mathbf{W_l}^k \overrightarrow{h_i}$$
        # $$\overrightarrow{{g_r}^k_i} = \mathbf{W_r}^k \overrightarrow{h_i}$$
        # for each head.
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)

        # #### Calculate attention score
        #
        # We calculate these for each head $k$. *We have omitted $\cdot^k$ for simplicity*.
        #
        # $$e_{ij} = a(\mathbf{W_l} \overrightarrow{h_i}, \mathbf{W_r} \overrightarrow{h_j}) =
        # a(\overrightarrow{{g_l}_i}, \overrightarrow{{g_r}_j})$$
        #
        # $e_{ij}$ is the attention score (importance) from node $j$ to node $i$.
        # We calculate this for each head.
        #
        # $a$ is the attention mechanism, that calculates the attention score.
        # The paper sums
        # $\overrightarrow{{g_l}_i}$, $\overrightarrow{{g_r}_j}$
        # followed by a $\text{LeakyReLU}$
        # and does a linear transformation with a weight vector $\mathbf{a} \in \mathbb{R}^{F'}$
        #
        #
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big(
        # \Big[
        # \overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}
        # \Big] \Big)$$
        # Note: The paper desrcibes $e_{ij}$ as
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big( \mathbf{W}
        # \Big[
        # \overrightarrow{h_i} \Vert \overrightarrow{h_j}
        # \Big] \Big)$$
        # which is equivalent to the definition we use here.

        # First we calculate
        # $\Big[\overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j} \Big]$
        # for all pairs of $i, j$.
        #
        # `g_l_repeat` gets
        # $$\{\overrightarrow{{g_l}_1}, \overrightarrow{{g_l}_2}, \dots, \overrightarrow{{g_l}_N},
        # \overrightarrow{{g_l}_1}, \overrightarrow{{g_l}_2}, \dots, \overrightarrow{{g_l}_N}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        # `g_r_repeat_interleave` gets
        # $$\{\overrightarrow{{g_r}_1}, \overrightarrow{{g_r}_1}, \dots, \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_r}_2}, \overrightarrow{{g_r}_2}, \dots, \overrightarrow{{g_r}_2}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        # Now we add the two tensors to get
        # $$\{\overrightarrow{{g_l}_1} + \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_l}_1} + \overrightarrow{{g_r}_2},
        # \dots, \overrightarrow{{g_l}_1}  +\overrightarrow{{g_r}_N},
        # \overrightarrow{{g_l}_2} + \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_l}_2} + \overrightarrow{{g_r}_2},
        # \dots, \overrightarrow{{g_l}_2}  + \overrightarrow{{g_r}_N}, ...\}$$
        g_sum = g_l_repeat + g_r_repeat_interleave
        # Reshape so that `g_sum[i, j]` is $\overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}$
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # Calculate
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big(
        # \Big[
        # \overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}
        # \Big] \Big)$$
        # `e` is of shape `[n_nodes, n_nodes, n_heads, 1]`
        e = self.attn(self.activation(g_sum))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # We then normalize attention scores (or coefficients)
        # $$\alpha_{ij} = \text{softmax}_j(e_{ij}) =
        # \frac{\exp(e_{ij})}{\sum_{j' \in \mathcal{N}_i} \exp(e_{ij'})}$$
        #
        # where $\mathcal{N}_i$ is the set of nodes connected to $i$.
        #
        # We do this by setting unconnected $e_{ij}$ to $- \infty$ which
        # makes $\exp(e_{ij}) \sim 0$ for unconnected pairs.
        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        # $$\overrightarrow{h'^k_i} = \sum_{j \in \mathcal{N}_i} \alpha^k_{ij} \overrightarrow{{g_r}_{j,k}}$$
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        # Concatenate the heads
        if self.is_concat:
            # $$\overrightarrow{h'_i} = \Bigg\Vert_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            # $$\overrightarrow{h'_i} = \frac{1}{K} \sum_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.mean(dim=1)



