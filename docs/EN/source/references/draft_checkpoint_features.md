# DFlash2 target features

DFlash2 concatenates the target hidden states selected by `target_layer_ids`
(or `dflash_config.target_layer_ids`). Indices are zero-based and follow
execution order, including repeated execution of shared Loop layers.

## Final normalized hidden state

Checkpoints trained with the target's final normalized hidden state declare
it as the last entry in `target_feature_sources`:

```json
{
  "target_layer_ids": [3, 27, 51, 54],
  "target_feature_sources": [3, 27, 51, 54, "final_postnorm"]
}
```

This configuration concatenates five hidden states. For a target hidden size
of 4096, `fc.weight` has shape `[4096, 20480]`. The collector applies the target's
final normalization to the final hidden state and appends it after the
intermediate features. Missing final hidden states and inconsistent feature
ordering raise errors.

For backward compatibility, the same declaration is accepted in
`loop_teacher.feature_sources`. If both fields are present, they must agree.
