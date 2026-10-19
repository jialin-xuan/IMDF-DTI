"""Defaults preserved from the supplied project; see REPRODUCIBILITY.md."""


class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-4
        self.Epoch = 200
        self.Batch_size = 128
        self.Patience = 40
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 50
        self.char_dim = 64
        self.loss_epsilon = 1.0
        self.lambda_l1 = 0.002
        self.warmup_epochs = 20
        self.lambda_balance = 0.01
        self.gate_lr_ratio = 0.1
        self.gate_update_interval = 5
        self.gate_freeze_epochs = 10

        self.use_xmask = True
        self.use_dfgu = True
        self.use_rescal_hybrid = True
        self.use_staged_optimization = True
