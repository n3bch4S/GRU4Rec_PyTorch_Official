import math
import numpy as np
import pandas as pd
import torch
from torch import autograd, nn
from torch.autograd import Variable
from collections import OrderedDict
import time

from torch.optim import Optimizer


class IndexedAdagradM(Optimizer):

    def __init__(self, params, lr=0.05, momentum=0.0, eps=1e-6):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super(IndexedAdagradM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"] = torch.full_like(
                    p, 0, memory_format=torch.preserve_format
                )
                if momentum > 0:
                    state["mom"] = torch.full_like(
                        p, 0, memory_format=torch.preserve_format
                    )

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"].share_memory_()
                if group["momentum"] > 0:
                    state["mom"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                clr = group["lr"]
                momentum = group["momentum"]
                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()
                    accs = state["acc"][grad_indices] + grad_values.pow(2)
                    state["acc"].index_copy_(0, grad_indices, accs)
                    accs.add_(group["eps"]).sqrt_().mul_(-1 / clr)
                    if momentum > 0:
                        moma = state["mom"][grad_indices]
                        moma.mul_(momentum).add_(grad_values / accs)
                        state["mom"].index_copy_(0, grad_indices, moma)
                        p.index_add_(0, grad_indices, moma)
                    else:
                        p.index_add_(0, grad_indices, grad_values / accs)
                else:
                    state["acc"].add_(grad.pow(2))
                    accs = state["acc"].add(group["eps"])
                    accs.sqrt_()
                    if momentum > 0:
                        mom = state["mom"]
                        mom.mul_(momentum).addcdiv_(grad, accs, value=-clr)
                        p.add_(mom)
                    else:
                        p.addcdiv_(grad, accs, value=-clr)
        return loss


def init_parameter_matrix(
    tensor: torch.Tensor, dim0_scale: int = 1, dim1_scale: int = 1
):
    sigma = math.sqrt(
        6.0 / float(tensor.size(0) / dim0_scale + tensor.size(1) / dim1_scale)
    )
    return nn.init._no_grad_uniform_(tensor, -sigma, sigma)


class GRUEmbedding(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GRUEmbedding, self).__init__()
        self.Wx0 = nn.Embedding(dim_in, dim_out * 3, sparse=True)
        self.Wrz0 = nn.Parameter(torch.empty((dim_out, dim_out * 2), dtype=torch.float))
        self.Wh0 = nn.Parameter(torch.empty((dim_out, dim_out * 1), dtype=torch.float))
        self.Bh0 = nn.Parameter(torch.zeros(dim_out * 3, dtype=torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        init_parameter_matrix(self.Wx0.weight, dim1_scale=3)
        init_parameter_matrix(self.Wrz0, dim1_scale=2)
        init_parameter_matrix(self.Wh0, dim1_scale=1)
        nn.init.zeros_(self.Bh0)

    def forward(self, X, H):
        Vx = self.Wx0(X) + self.Bh0
        Vrz = torch.mm(H, self.Wrz0)
        vx_x, vx_r, vx_z = Vx.chunk(3, 1)
        vh_r, vh_z = Vrz.chunk(2, 1)
        r = torch.sigmoid(vx_r + vh_r)
        z = torch.sigmoid(vx_z + vh_z)
        h = torch.tanh(torch.mm(r * H, self.Wh0) + vx_x)
        h = (1.0 - z) * H + z * h
        return h


class GRU4RecModel(nn.Module):
    def __init__(
        self,
        n_items,  # Number of unique items in the dataset
        layers=[
            100
        ],  # List of hidden layer sizes (e.g., [100] for a single layer with 100 units)
        dropout_p_embed=0.0,  # Dropout probability for the embedding layer
        dropout_p_hidden=0.0,  # Dropout probability for hidden layers
        embedding=0,  # Dimension of the item embeddings, 0 if not using explicit embeddings
        constrained_embedding=True,  # Whether to use a constrained embedding setup
    ):
        super(GRU4RecModel, self).__init__()

        # Initialize parameters
        self.n_items = n_items  # Total number of items
        self.layers = layers  # List of hidden layer dimensions
        self.dropout_p_embed = dropout_p_embed  # Dropout probability for embeddings
        self.dropout_p_hidden = (
            dropout_p_hidden  # Dropout probability for hidden layers
        )
        self.embedding = embedding  # Size of the embedding vectors
        self.constrained_embedding = (
            constrained_embedding  # Toggle for constrained embedding
        )
        self.start = 0  # Indicator of where to start GRU layers

        # Determine input size based on embedding settings
        if constrained_embedding:
            # Use the last hidden layer's size as the input when constrained embeddings are used
            n_input = layers[-1]
        elif embedding:
            # Initialize explicit embedding layer if embedding size is provided
            self.E = nn.Embedding(n_items, embedding, sparse=True)
            n_input = embedding
        else:
            # Use a custom GRUEmbedding when no explicit embedding is provided
            self.GE = GRUEmbedding(n_items, layers[0])
            n_input = n_items
            self.start = 1  # Adjust start index when using GRUEmbedding

        # Dropout layer for embeddings
        self.DE = nn.Dropout(dropout_p_embed)

        # Initialize GRU cells and Dropout layers for hidden layers
        self.G = []  # List to store GRU cells
        self.D = []  # List to store dropout layers
        for i in range(self.start, len(layers)):
            # Append a GRUCell for each hidden layer
            self.G.append(nn.GRUCell(layers[i - 1] if i > 0 else n_input, layers[i]))

            # Append a dropout layer for each hidden layer
            self.D.append(nn.Dropout(dropout_p_hidden))

        # Convert lists to ModuleLists for use in nn.Module
        self.G = nn.ModuleList(self.G)
        self.D = nn.ModuleList(self.D)

        # Embedding layers for the output (used for final scoring)
        self.Wy = nn.Embedding(
            n_items, layers[-1], sparse=True
        )  # Weight matrix for items
        self.By = nn.Embedding(n_items, 1, sparse=True)  # Bias vector for items

        # Initialize parameters of the model
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.embedding:
            init_parameter_matrix(self.E.weight)
        elif not self.constrained_embedding:
            self.GE.reset_parameters()
        for i in range(len(self.G)):
            init_parameter_matrix(self.G[i].weight_ih, dim1_scale=3)
            init_parameter_matrix(self.G[i].weight_hh, dim1_scale=3)
            nn.init.zeros_(self.G[i].bias_ih)
            nn.init.zeros_(self.G[i].bias_hh)
        init_parameter_matrix(self.Wy.weight)
        nn.init.zeros_(self.By.weight)

    def _init_numpy_weights(self, shape):
        sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
        m = np.random.rand(*shape).astype("float32") * 2 * sigma - sigma
        return m

    @torch.no_grad()
    def _reset_weights_to_compatibility_mode(self):
        np.random.seed(42)
        if self.constrained_embedding:
            n_input = self.layers[-1]
        elif self.embedding:
            n_input = self.embedding
            self.E.weight.set_(
                torch.tensor(
                    self._init_numpy_weights((self.n_items, n_input)),
                    device=self.E.weight.device,
                )
            )
        else:
            n_input = self.n_items
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            self.GE.Wx0.weight.set_(
                torch.tensor(np.hstack(m), device=self.GE.Wx0.weight.device)
            )
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[0], self.layers[0])))
            m2.append(self._init_numpy_weights((self.layers[0], self.layers[0])))
            self.GE.Wrz0.set_(torch.tensor(np.hstack(m2), device=self.GE.Wrz0.device))
            self.GE.Wh0.set_(
                torch.tensor(
                    self._init_numpy_weights((self.layers[0], self.layers[0])),
                    device=self.GE.Wh0.device,
                )
            )
            self.GE.Bh0.set_(
                torch.zeros((self.layers[0] * 3,), device=self.GE.Bh0.device)
            )
        for i in range(self.start, len(self.layers)):
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            self.G[i].weight_ih.set_(
                torch.tensor(np.vstack(m), device=self.G[i].weight_ih.device)
            )
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            self.G[i].weight_hh.set_(
                torch.tensor(np.vstack(m2), device=self.G[i].weight_hh.device)
            )
            self.G[i].bias_hh.set_(
                torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_hh.device)
            )
            self.G[i].bias_ih.set_(
                torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_ih.device)
            )
        self.Wy.weight.set_(
            torch.tensor(
                self._init_numpy_weights((self.n_items, self.layers[-1])),
                device=self.Wy.weight.device,
            )
        )
        self.By.weight.set_(
            torch.zeros((self.n_items, 1), device=self.By.weight.device)
        )

    def embed_constrained(self, X, Y=None):
        if Y is not None:
            XY = torch.cat([X, Y])
            EXY = self.Wy(XY)
            split = X.shape[0]
            E = EXY[:split]
            O = EXY[split:]
            B = self.By(Y)
        else:
            E = self.Wy(X)
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    # Handles embedding generation when separate embedding layers are used for input and output
    def embed_separate(self, X, Y=None):
        E = self.E(X)  # Embedding for the input items
        if Y is not None:  # If target items are provided
            O = self.Wy(Y)  # Output weight embeddings for target items
            B = self.By(Y)  # Output bias embeddings for target items
        else:  # If no target items are provided, use all items
            O = self.Wy.weight  # Output weights for all items
            B = self.By.weight  # Output biases for all items
        return E, O, B

    def embed_gru(self, X, H, Y=None):
        E = self.GE(X, H)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    # Selects the appropriate embedding generation method based on the model's configuration
    def embed(self, X, H, Y=None):
        if self.constrained_embedding:
            # Use constrained embedding (input size matches last hidden layer size)
            E, O, B = self.embed_constrained(X, Y)
        elif self.embedding > 0:
            # Use separate embedding layers for input and output
            E, O, B = self.embed_separate(X, Y)
        else:
            # Use GRUEmbedding when no explicit embedding layers are used
            E, O, B = self.embed_gru(X, H[0], Y)
        return E, O, B

    # Executes a single step of the GRU hidden state update
    def hidden_step(self, X, H, training=False):
        for i in range(self.start, len(self.layers)):
            X = self.G[i](X, Variable(H[i]))  # Update the GRU hidden state
            if training:  # Apply dropout during training
                X = self.D[i](X)
            H[i] = X  # Update the hidden state
        return X  # Return the final hidden state

    # Computes scores for all items based on the current hidden state and output embeddings
    def score_items(self, X, O, B):
        O = torch.mm(X, O.T) + B.T  # Linear transformation (dot product) + bias
        return O

    # Defines the forward pass of the model
    def forward(self, X, H, Y, training=False):
        # Generate embeddings for the input, output weights, and biases
        E, O, B = self.embed(X, H, Y)
        if training:
            E = self.DE(E)  # Apply dropout to embeddings during training
        if not (self.constrained_embedding or self.embedding):
            H[0] = (
                E  # Use embeddings as the initial hidden state if no explicit embeddings are used
            )
        Xh = self.hidden_step(E, H, training=training)  # Update the hidden state
        R = self.score_items(Xh, O, B)  # Compute item scores
        return R


class SampleCache:
    def __init__(
        self, n_sample, sample_cache_max_size, distr, device=torch.device("cuda:0")
    ):
        self.device = device
        self.n_sample = n_sample
        self.generate_length = sample_cache_max_size // n_sample if n_sample > 0 else 0
        self.distr = distr
        self._refresh()
        print(
            "Created sample store with {} batches of samples (type=GPU)".format(
                self.generate_length
            )
        )

    def _bin_search(self, arr, x):
        l = x.shape[0]
        a = torch.zeros(l, dtype=torch.int64, device=self.device)
        b = torch.zeros(l, dtype=torch.int64, device=self.device) + arr.shape[0]
        while torch.any(a != b):
            ab = torch.div((a + b), 2, rounding_mode="trunc")
            val = arr[ab]
            amask = val <= x
            a[amask] = ab[amask] + 1
            b[~amask] = ab[~amask]
        return a

    def _refresh(self):
        if self.n_sample <= 0:
            return
        x = torch.rand(
            self.generate_length * self.n_sample,
            dtype=torch.float32,
            device=self.device,
        )
        self.neg_samples = self._bin_search(self.distr, x).reshape(
            (self.generate_length, self.n_sample)
        )
        self.sample_pointer = 0

    def get_sample(self):
        if self.sample_pointer >= self.generate_length:
            self._refresh()
        sample = self.neg_samples[self.sample_pointer]
        self.sample_pointer += 1
        return sample


class SessionDataIterator:
    def __init__(
        self,
        data,
        batch_size,
        n_sample=0,
        sample_alpha=0.75,
        sample_cache_max_size=10000000,
        item_key="ItemId",
        session_key="SessionId",
        time_key="Time",
        session_order="time",
        device=torch.device("cuda:0"),
        itemidmap=None,
    ):
        self.device = device
        self.batch_size = batch_size
        if itemidmap is None:
            itemids = data[item_key].unique()
            self.n_items = len(itemids)
            self.itemidmap = pd.Series(
                data=np.arange(self.n_items, dtype="int32"),
                index=itemids,
                name="ItemIdx",
            )
        else:
            print("Using existing item ID map")
            self.itemidmap = itemidmap
            self.n_items = len(itemidmap)
            in_mask = data[item_key].isin(itemidmap.index.values)
            n_not_in = (~in_mask).sum()
            if n_not_in > 0:
                # print('{} rows of the data contain unknown items and will be filtered'.format(n_not_in))
                data = data.drop(data.index[~in_mask])
        self.sort_if_needed(data, [session_key, time_key])
        self.offset_sessions = self.compute_offset(data, session_key)
        if session_order == "time":
            self.session_idx_arr = np.argsort(
                data.groupby(session_key)[time_key].min().values
            )
        else:
            self.session_idx_arr = np.arange(len(self.offset_sessions) - 1)
        self.data_items = self.itemidmap[data[item_key].values].values
        if n_sample > 0:
            pop = data.groupby(item_key).size()
            pop = pop[self.itemidmap.index.values].values ** sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1
            distr = torch.tensor(pop, device=self.device, dtype=torch.float32)
            self.sample_cache = SampleCache(
                n_sample, sample_cache_max_size, distr, device=self.device
            )

    def sort_if_needed(self, data, columns, any_order_first_dim=False):
        is_sorted = True
        neq_masks = []
        for i, col in enumerate(columns):
            dcol = data[col]
            neq_masks.append(dcol.values[1:] != dcol.values[:-1])
            if i == 0:
                if any_order_first_dim:
                    is_sorted = is_sorted and (dcol.nunique() == neq_masks[0].sum() + 1)
                else:
                    is_sorted = is_sorted and np.all(
                        dcol.values[1:] >= dcol.values[:-1]
                    )
            else:
                is_sorted = is_sorted and np.all(
                    neq_masks[i - 1] | (dcol.values[1:] >= dcol.values[:-1])
                )
            if not is_sorted:
                break
        if is_sorted:
            print("The dataframe is already sorted by {}".format(", ".join(columns)))
        else:
            print("The dataframe is not sorted by {}, sorting now".format(col))
            t0 = time.time()
            data.sort_values(columns, inplace=True)
            t1 = time.time()
            print("Data is sorted in {:.2f}".format(t1 - t0))

    def compute_offset(self, data, column):
        offset = np.zeros(data[column].nunique() + 1, dtype=np.int32)
        offset[1:] = data.groupby(column).size().cumsum()
        return offset

    def __call__(self, enable_neg_samples, reset_hook=None):
        batch_size = self.batch_size
        iters = np.arange(batch_size)
        maxiter = iters.max()
        start = self.offset_sessions[self.session_idx_arr[iters]]
        end = self.offset_sessions[self.session_idx_arr[iters] + 1]
        finished = False
        valid_mask = np.ones(batch_size, dtype="bool")
        n_valid = self.batch_size
        while not finished:
            minlen = (end - start).min()
            out_idx = torch.tensor(
                self.data_items[start], requires_grad=False, device=self.device
            )
            for i in range(minlen - 1):
                in_idx = out_idx
                out_idx = torch.tensor(
                    self.data_items[start + i + 1],
                    requires_grad=False,
                    device=self.device,
                )
                if enable_neg_samples:
                    sample = self.sample_cache.get_sample()
                    y = torch.cat([out_idx, sample])
                else:
                    y = out_idx
                yield in_idx, y
            start = start + minlen - 1
            finished_mask = end - start <= 1
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
            maxiter += n_finished
            valid_mask = iters < len(self.offset_sessions) - 1
            n_valid = valid_mask.sum()
            if n_valid == 0:
                finished = True
                break
            mask = finished_mask & valid_mask
            sessions = self.session_idx_arr[iters[mask]]
            start[mask] = self.offset_sessions[sessions]
            end[mask] = self.offset_sessions[sessions + 1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
            if reset_hook is not None:
                finished = reset_hook(n_valid, finished_mask, valid_mask)


class GRU4Rec:
    def __init__(
        self,
        layers=[100],
        loss="cross-entropy",
        batch_size=64,
        dropout_p_embed=0.0,
        dropout_p_hidden=0.0,
        learning_rate=0.05,
        momentum=0.0,
        sample_alpha=0.5,
        n_sample=2048,
        embedding=0,
        constrained_embedding=True,
        n_epochs=10,
        bpreg=1.0,
        elu_param=0.5,
        logq=0.0,
        device=torch.device("cuda:0"),
    ):
        self.device = device
        self.layers = layers
        self.loss = loss
        self.set_loss_function(loss)
        self.elu_param = elu_param
        self.bpreg = bpreg
        self.logq = logq
        self.batch_size = batch_size
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sample_alpha = sample_alpha
        self.n_sample = n_sample
        if embedding == "layersize":
            self.embedding = self.layers[0]
        else:
            self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.n_epochs = n_epochs

    def set_loss_function(self, loss):
        if loss == "cross-entropy":
            self.loss_function = self.xe_loss_with_softmax
        elif loss == "bpr-max":
            self.loss_function = self.bpr_max_loss_with_elu
        elif loss == "top1":
            self.loss_function = self.top1_loss
        else:
            raise NotImplementedError

    def set_params(self, **kvargs):
        maxk_len = np.max([len(str(x)) for x in kvargs.keys()])
        maxv_len = np.max([len(str(x)) for x in kvargs.values()])
        for k, v in kvargs.items():
            if not hasattr(self, k):
                print("Unkown attribute: {}".format(k))
                raise NotImplementedError
            else:
                if type(v) == str and type(getattr(self, k)) == list:
                    v = [int(l) for l in v.split("/")]
                if type(v) == str and type(getattr(self, k)) == bool:
                    if v == "True" or v == "1":
                        v = True
                    elif v == "False" or v == "0":
                        v = False
                    else:
                        print("Invalid value for boolean parameter: {}".format(v))
                        raise NotImplementedError
                if k == "embedding" and v == "layersize":
                    self.embedding = "layersize"
                setattr(self, k, type(getattr(self, k))(v))
                if k == "loss":
                    self.set_loss_function(self.loss)
                print(
                    "SET   {}{}TO   {}{}(type: {})".format(
                        k,
                        " " * (maxk_len - len(k) + 3),
                        getattr(self, k),
                        " " * (maxv_len - len(str(getattr(self, k))) + 3),
                        type(getattr(self, k)),
                    )
                )
        if self.embedding == "layersize":
            self.embedding = self.layers[0]
            print(
                "SET   {}{}TO   {}{}(type: {})".format(
                    "embedding",
                    " " * (maxk_len - len("embedding") + 3),
                    getattr(self, "embedding"),
                    " " * (maxv_len - len(str(getattr(self, "embedding"))) + 3),
                    type(getattr(self, "embedding")),
                )
            )

    # Cross-entropy loss with softmax adjustment
    def xe_loss_with_softmax(self, O, Y, M):
        if self.logq > 0:
            # Adjust the scores using log probabilities of the items
            O = O - self.logq * torch.log(
                torch.cat([self.P0[Y[:M]], self.P0[Y[M:]] ** self.sample_alpha])
            )
        # Compute softmax scores for the output logits
        X = torch.exp(O - O.max(dim=1, keepdim=True)[0])  # Numerical stability trick
        X = X / X.sum(dim=1, keepdim=True)  # Normalize to get probabilities
        # Calculate negative log-likelihood for the correct items
        return -torch.sum(torch.log(torch.diag(X) + 1e-24))

    # Helper function to compute softmax for negative sampling
    def softmax_neg(self, X):
        # Create a mask to ignore diagonal elements (target items)
        hm = 1.0 - torch.eye(*X.shape, out=torch.empty_like(X))
        X = X * hm  # Zero out diagonal elements
        # Compute softmax probabilities for negative items
        e_x = torch.exp(X - X.max(dim=1, keepdim=True)[0]) * hm
        return e_x / e_x.sum(dim=1, keepdim=True)  # Normalize across each row

    # BPR-max loss with optional ELU activation
    def bpr_max_loss_with_elu(self, O, Y, M):
        if self.elu_param > 0:
            # Apply ELU activation to logits
            O = nn.functional.elu(O, self.elu_param)
        # Compute softmax probabilities for negative items
        softmax_scores = self.softmax_neg(O)
        # Extract scores for the target items (diagonal of O)
        target_scores = torch.diag(O)
        target_scores = target_scores.reshape(target_scores.shape[0], -1)
        # Calculate the BPR-max loss
        return torch.sum(
            (
                -torch.log(
                    torch.sum(torch.sigmoid(target_scores - O) * softmax_scores, dim=1)
                    + 1e-24
                )  # BPR-max term
                + self.bpreg
                * torch.sum((O**2) * softmax_scores, dim=1)  # Regularization term
            )
        )

    # Top-1 loss with optional ELU activation
    def top1_loss(self, O, Y, M):
        if self.elu_param > 0:
            # Apply ELU activation to logits
            O = nn.functional.elu(O, self.elu_param)

        # The target scores are the diagonal elements of O (correct items' scores)
        target_scores = torch.diag(O)
        target_scores = target_scores.reshape(target_scores.shape[0], -1)

        # Compute the sigmoid difference between target scores and other scores
        sigmoid_diff = torch.sigmoid(O - target_scores)

        # Calculate the Top-1 Loss
        loss = torch.sum(
            torch.sum(sigmoid_diff, dim=1)  # Sum of differences for all incorrect items
            + torch.sum(
                torch.sigmoid(O**2) * torch.sigmoid(O), dim=1
            )  # Regularization term
        )

        return loss

    def fit(
        self,
        data,
        sample_cache_max_size=10000000,  # Maximum size for caching samples
        compatibility_mode=True,  # Whether to use weights compatible with older versions
        item_key="ItemId",  # Key for item IDs in the dataset
        session_key="SessionId",  # Key for session IDs in the dataset
        time_key="Time",  # Key for timestamps in the dataset
    ):
        # Track if any error occurs during training
        self.error_during_train = False

        # Initialize the data iterator for session-based recommendation
        self.data_iterator = SessionDataIterator(
            data,
            self.batch_size,  # Batch size for training
            n_sample=self.n_sample,  # Number of negative samples
            sample_alpha=self.sample_alpha,  # Sampling distribution parameter
            sample_cache_max_size=sample_cache_max_size,  # Limit for sampling cache size
            item_key=item_key,
            session_key=session_key,
            time_key=time_key,
            session_order="time",  # Order sessions by time for training
            device=self.device,  # Move data to the specified device (e.g., GPU)
        )

        # Precompute item popularity for cross-entropy loss if logq is set
        if self.logq and self.loss == "cross-entropy":
            pop = data.groupby(item_key).size()  # Get popularity of each item
            self.P0 = torch.tensor(
                pop[self.data_iterator.itemidmap.index.values],
                dtype=torch.float32,
                device=self.device,
            )

        # Initialize the GRU model
        model = GRU4RecModel(
            self.data_iterator.n_items,  # Number of unique items
            self.layers,  # GRU hidden layer sizes
            self.dropout_p_embed,  # Dropout probability for embeddings
            self.dropout_p_hidden,  # Dropout probability for hidden layers
            self.embedding,  # Embedding dimension
            self.constrained_embedding,  # Whether embeddings are constrained
        ).to(
            self.device
        )  # Move the model to the specified device

        # Apply compatibility mode for weight initialization if needed
        if compatibility_mode:
            model._reset_weights_to_compatibility_mode()

        # Store the model in the GRU4Rec object
        self.model = model

        # Initialize the optimizer with IndexedAdagradM
        opt = IndexedAdagradM(
            self.model.parameters(),  # Parameters to optimize
            self.learning_rate,  # Learning rate
            self.momentum,  # Momentum for the optimizer
        )

        # Training loop over epochs
        for epoch in range(self.n_epochs):
            t0 = time.time()  # Start time of the epoch

            # Initialize hidden states for the GRU layers
            H = []
            for i in range(len(self.layers)):
                H.append(
                    torch.zeros(
                        (
                            self.batch_size,
                            self.layers[i],
                        ),  # Shape: (batch_size, layer_size)
                        dtype=torch.float32,
                        requires_grad=False,
                        device=self.device,  # Allocate on the same device as the model
                    )
                )

            c = []  # Store loss values
            cc = []  # Store counts of valid samples
            n_valid = self.batch_size  # Number of valid samples in each batch

            # Define a hook to adjust hidden states when sessions finish or reset
            reset_hook = lambda n_valid, finished_mask, valid_mask: self._adjust_hidden(
                n_valid, finished_mask, valid_mask, H
            )

            # Iterate over batches of input and output indices
            for in_idx, out_idx in self.data_iterator(
                enable_neg_samples=(self.n_sample > 0),  # Enable negative sampling
                reset_hook=reset_hook,  # Hook for resetting hidden states
            ):
                for h in H:
                    h.detach_()  # Detach hidden states to avoid gradient accumulation

                self.model.zero_grad()  # Zero out gradients from the previous step

                # Forward pass through the model
                R = self.model.forward(in_idx, H, out_idx, training=True)

                # Compute the loss
                L = self.loss_function(R, out_idx, n_valid) / self.batch_size
                L.backward()  # Backpropagate the gradients
                opt.step()  # Update model parameters

                # Store the loss value
                L = L.cpu().detach().numpy()
                c.append(L)
                cc.append(n_valid)

                # Handle NaN errors in the loss
                if np.isnan(L):
                    print(str(epoch) + ": NaN error!")
                    self.error_during_train = True
                    return

            # Calculate the average loss for the epoch
            c = np.array(c)
            cc = np.array(cc)
            avgc = np.sum(c * cc) / np.sum(cc)

            # Check for NaN errors in the average loss
            if np.isnan(avgc):
                print("Epoch {}: NaN error!".format(str(epoch)))
                self.error_during_train = True
                return

            # Calculate elapsed time for the epoch
            t1 = time.time()
            dt = t1 - t0

            # Print progress and performance metrics
            print(
                "Epoch{} --> loss: {:.6f} \t({:.2f}s) \t[{:.2f} mb/s | {:.0f} e/s]".format(
                    epoch + 1, avgc, dt, len(c) / dt, np.sum(cc) / dt
                )
            )

    def _adjust_hidden(self, n_valid, finished_mask, valid_mask, H):
        if (self.n_sample == 0) and (n_valid < 2):
            return True
        with torch.no_grad():
            for i in range(len(self.layers)):
                H[i][finished_mask] = 0
        if n_valid < len(valid_mask):
            for i in range(len(H)):
                H[i] = H[i][valid_mask]
        return False

    def to(self, device):
        if type(device) == str:
            device = torch.device(device)
        if device == self.device:
            return
        if hasattr(self, "model"):
            self.model = self.model.to(device)
            self.model.eval()
        self.device = device
        if hasattr(self, "data_iterator"):
            self.data_iterator.device = device
            if hasattr(self.data_iterator, "sample_cache"):
                self.data_iterator.sample_cache.device = device
        pass

    def savemodel(self, path):
        torch.save(self, path)

    @classmethod
    def loadmodel(cls, path, device="cuda:0"):
        gru = torch.load(path, map_location=device)
        gru.device = torch.device(device)
        if hasattr(gru, "data_iterator"):
            gru.data_iterator.device = torch.device(device)
            if hasattr(gru.data_iterator, "sample_cache"):
                gru.data_iterator.sample_cache.device = torch.device(device)
        gru.model.eval()
        return gru
