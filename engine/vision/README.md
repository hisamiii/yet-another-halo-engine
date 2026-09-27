# vision/

`mmproj-F16.gguf` (0.86 GiB) is the only projector.

- Image preprocessing and the projector, from the target's own metadata.
- Image tokens merge after the projector, 64-16384 per image.
- Routing the projector to the NPU is unmeasured. Do not assume it behaves like text prefill; a ViT has different shapes and the xclbins are shape-baked.

Gate: image prompt produces correct output.
