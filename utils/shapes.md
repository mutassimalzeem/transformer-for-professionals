# Common Shape Notes

## Symbols
- B = batch size
- T = sequence length
- d_model = embedding dimension
- h = number of heads
- d_k = key dimension
- d_v = value dimension

## Common Shapes

### Embedding output
(B, T, d_model)

### Positional encoding
(T, d_model) or (1, T, d_model)

### Q, K, V
(B, T, d_model)

### Attention score matrix
(B, T, T)

### Multi-head split
(B, h, T, d_k)

### Multi-head output after concat
(B, T, d_model)