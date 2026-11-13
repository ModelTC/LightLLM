"""Controlled Full Attention timing reference on the same step519 weights.

Only private benchmark snapshots load this module. It removes DSA from the
normalized runtime configuration, while retaining the baseline YaRN cache.
Its outputs are deliberately not a DSA precision reference.
"""


def install(model_class):
    normalize = model_class.normalize_model_config.__func__

    def dense_config(cls, model_config):
        config = dict(model_config)
        config.pop("dsa", None)
        config.pop("dsa_export", None)
        return normalize(cls, config)

    def yarn(self):
        from .dsa import yarn_cos_sin

        self._cos_cached, self._sin_cached = yarn_cos_sin(self.config, self.max_seq_length, "cuda")

    model_class.normalize_model_config = classmethod(dense_config)
    model_class._init_to_get_yarn_rotary = yarn
