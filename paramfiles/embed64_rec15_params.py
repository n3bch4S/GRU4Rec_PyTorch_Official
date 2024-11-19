from collections import OrderedDict

gru4rec_params = OrderedDict(
    [
        ("loss", "top1"),
        ("constrained_embedding", False),
        # TODO: set embedding size as assignment told
        ("embedding", 64),
        ("elu_param", 0),
        # TODO: set along with embedding
        ("layers", [64]),
        ("n_epochs", 10),
        ("batch_size", 50),
        ("dropout_p_embed", 0.0),
        ("dropout_p_hidden", 0.5),
        ("learning_rate", 0.01),
        ("momentum", 0),
        # TODO: try to find best n_sample
        ("n_sample", 32),
        ("sample_alpha", 0.5),
        ("bpreg", 0.0),
        ("logq", 1.0),
    ]
)
