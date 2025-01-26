import torch
import torch.nn as nn



# LayerNorm class
class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = 1e-5


    # pass the output of the first layer to this function
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        out_norm = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * out_norm + self.bias
    




# GELU activation function
class Gelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x**3)))
    


# FeedForward class
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embedding_dim"], cfg["embedding_dim"] * 4),
            Gelu(),
            nn.Linear(cfg["embedding_dim"] * 4, cfg["embedding_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    


# Residual class - shortcut connection
class Residual(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut

        # 5 layers
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), Gelu()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), Gelu()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), Gelu()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), Gelu()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), Gelu())

        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                # add the input to the output
                x = x + layer_output
            else:
                x = layer_output
        return x
    


# MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_in, d_out, context_length, qkv_bias=False, dropout=0.0):
        super().__init__()
        # creating a list of single head self attention layers
        assert (d_out % num_heads) == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # dimension of each head

        # setting weights for query, key, and value for all heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # combining head outputs
        self.out_projection = nn.Linear(d_out, d_out)

        # adding dropout layer
        self.dropout = nn.Dropout(dropout)

        # creating buffer for mask
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, X):
        batch, num_tokens, d_in = X.shape #3D vector

        # computing query, key, and value vectors
        query = self.W_query(X)
        key = self.W_key(X)
        value = self.W_value(X)

        # splitting query, key, and value vectors for all heads - changing dimensions
        query = query.view(batch, num_tokens, self.num_heads, self.head_dim)
        key = key.view(batch, num_tokens, self.num_heads, self.head_dim)
        value = value.view(batch, num_tokens, self.num_heads, self.head_dim)

        # transposing to get the right dimensions
        keys = key.transpose(1,2)
        queries = query.transpose(1,2)
        values = value.transpose(1,2)


        # computing attention scores
        attention_scores = queries @ keys.transpose(2,3)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # computing attention weights
        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)

        # applying dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        

        # computing context vectors
        context_vec = (attention_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out)
        

        # combining head outputs
        context_vec = self.out_projection(context_vec)
        
        return context_vec



# TransformerBlock class
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["embedding_dim"],
            d_out=cfg["embedding_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["dropout_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.drop_shortcut = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x




# GPTModel - combining all the classes
class GPT_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])] #12 transformer blocks
        )
        self.final_norm = LayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x) #applying dropout
        x = self.trf_blocks(x) #passing through transformer blocks
        x = self.final_norm(x) #normalizing the output
        logits = self.out_head(x) #next tokens unormalized probabilities
        return logits
    



# GPT function
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx