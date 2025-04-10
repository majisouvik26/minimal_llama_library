import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleLLM(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_weights(model_name)
        self.rms_norm_eps = getattr(self.config, 'rms_norm_eps', 1e-6)
        self._initialize_rope_cache()

    def _initialize_rope_cache(self, max_seq_len=2048, theta=10000.0):
        """
        This function initializes a cache for a rope data structure with a specified maximum sequence
        length and theta value.
        
        :param max_seq_len: The `max_seq_len` parameter specifies the maximum length of the sequence
        that will be stored in the rope cache. This parameter helps limit the memory usage of the cache
        by setting a maximum size for the sequences stored in it, defaults to 2048 (optional)
        :param theta: The `theta` parameter in the `_initialize_rope_cache` function is a value used for
        a specific purpose within the function. It seems to be a constant value that influences the
        behavior or calculations performed within the function. In this case, `theta` is set to a
        default value of `100
        """
        dim = self.head_dim
        positions = torch.arange(max_seq_len, device=self.device)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=self.device).float() / dim))
        emb = positions.unsqueeze(1) * freqs.unsqueeze(0) 
        self.cos_cached = torch.cos(emb) 
        self.sin_cached = torch.sin(emb)  

   
    def load_weights(self, model_name):
        """
    The function `load_weights` loads weights from a pre-trained model and initializes related
    attributes.
    
    :param model_name: The `model_name` parameter in the `load_weights` function is a string that
    represents the name of the pre-trained model that you want to load the weights from. This model name
    is used to initialize an instance of `AutoModelForCausalLM` and `AutoTokenizer` from the H
    """
        
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = model.config
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', self.num_attention_heads)
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    def rms_norm(self, x, weight):
        """
        The function `rms_norm` calculates the root mean square normalization of input `x` with given
        `weight`.
        
        :param x: The parameter `x` in the `rms_norm` function is typically a tensor representing the
        input data or features that you want to normalize using root mean square (RMS) normalization
        :param weight: The `weight` parameter in the `rms_norm` function is used to scale the normalized
        input `x`. It is multiplied with the normalized input before returning the final result. This
        allows you to apply a weight to the normalized input values as part of the normalization process
        :return: The function `rms_norm` returns the normalized input `x` multiplied by the weight
        `weight`.
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return x * weight

    def apply_rope(self, x, seq_len):
        """
        The function `apply_rope` takes an input tensor `x`, reshapes it, rotates its components using
        cached cosine and sine values, and concatenates the rotated components to form a new tensor
        `x_rope`.
        
        :param x: The `x` parameter in the `apply_rope` function represents an input tensor with shape
        `(batch_size, num_heads, seq_len, head_dim)`
        :param seq_len: The `seq_len` parameter in the `apply_rope` function represents the length of
        the sequence being processed. It is used to determine the range of cached cosine and sine values
        that are applied during the rotation operation on the input tensor `x`
        :return: The function `apply_rope` returns the tensor `x_rope`, which is the concatenation of
        `x_rotated_first` and `x_rotated_second` along the last dimension.
        """
        batch_size, num_heads, _, head_dim = x.shape
        x = x.view(batch_size, num_heads, seq_len, head_dim)
        
        half_dim = head_dim // 2
        x1 = x[..., :half_dim]  
        x2 = x[..., half_dim:]  
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        x_rotated_first = x1 * cos - x2 * sin
        x_rotated_second = x1 * sin + x2 * cos
        
        x_rope = torch.cat([x_rotated_first, x_rotated_second], dim=-1)
        
        return x_rope

    def forward(self, input_ids):
        """
        The function `forward` processes input data through multiple layers of a neural network model and
        returns the logits for language modeling.
        
        :param input_ids: The code you provided seems to be implementing a forward pass of a neural
        network model, possibly a transformer-based model for natural language processing. The `forward`
        method processes input_ids through multiple layers of the model to generate logits for language
        modeling
        :return: the logits after performing the forward pass through the transformer model. The logits
        are obtained by multiplying the final output tensor `x` with the `lm_head.weight` tensor and
        adding the bias term if it exists in the weights dictionary.
        """
        input_ids = input_ids.to(self.device)
        x = self.weights["model.embed_tokens.weight"][input_ids] 
        batch_size, seq_len, _ = x.shape

        for i in range(self.config.num_hidden_layers):
            ln1_weight = self.weights[f"model.layers.{i}.input_layernorm.weight"]
            x_norm = self.rms_norm(x, ln1_weight)
            q_weight = self.weights[f"model.layers.{i}.self_attn.q_proj.weight"]
            k_weight = self.weights[f"model.layers.{i}.self_attn.k_proj.weight"]
            v_weight = self.weights[f"model.layers.{i}.self_attn.v_proj.weight"]
            
            q = torch.matmul(x_norm, q_weight.T)
            k = torch.matmul(x_norm, k_weight.T)
            v = torch.matmul(x_norm, v_weight.T)
            
            if f"model.layers.{i}.self_attn.q_proj.bias" in self.weights:
                q = q + self.weights[f"model.layers.{i}.self_attn.q_proj.bias"]
            if f"model.layers.{i}.self_attn.k_proj.bias" in self.weights:
                k = k + self.weights[f"model.layers.{i}.self_attn.k_proj.bias"]
            if f"model.layers.{i}.self_attn.v_proj.bias" in self.weights:
                v = v + self.weights[f"model.layers.{i}.self_attn.v_proj.bias"]

            q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            q = self.apply_rope(q, seq_len)
            k = self.apply_rope(k, seq_len)

            if self.num_key_value_heads < self.num_attention_heads:
                heads_per_group = self.num_attention_heads // self.num_key_value_heads
                k = k.repeat_interleave(heads_per_group, dim=1)
                v = v.repeat_interleave(heads_per_group, dim=1)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
                diagonal=1
            )
            mask = mask.unsqueeze(0).unsqueeze(0) 
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            o_weight = self.weights[f"model.layers.{i}.self_attn.o_proj.weight"]
            attn_output = torch.matmul(attn_output, o_weight.T)
            
            if f"model.layers.{i}.self_attn.o_proj.bias" in self.weights:
                attn_output = attn_output + self.weights[f"model.layers.{i}.self_attn.o_proj.bias"]
            
            x = x + attn_output

            ln2_weight = self.weights[f"model.layers.{i}.post_attention_layernorm.weight"]
            x_norm2 = self.rms_norm(x, ln2_weight)
            
            gate_weight = self.weights[f"model.layers.{i}.mlp.gate_proj.weight"]
            up_weight = self.weights[f"model.layers.{i}.mlp.up_proj.weight"]
            
            gate = torch.matmul(x_norm2, gate_weight.T)
            up = torch.matmul(x_norm2, up_weight.T)
            
            if f"model.layers.{i}.mlp.gate_proj.bias" in self.weights:
                gate = gate + self.weights[f"model.layers.{i}.mlp.gate_proj.bias"]
            if f"model.layers.{i}.mlp.up_proj.bias" in self.weights:
                up = up + self.weights[f"model.layers.{i}.mlp.up_proj.bias"]
            
            gate = F.silu(gate)
            down_weight = self.weights[f"model.layers.{i}.mlp.down_proj.weight"]
            ff = torch.matmul(up * gate, down_weight.T)
            
            if f"model.layers.{i}.mlp.down_proj.bias" in self.weights:
                ff = ff + self.weights[f"model.layers.{i}.mlp.down_proj.bias"]
            
            x = x + ff

        norm_weight = self.weights["model.norm.weight"]
        x = self.rms_norm(x, norm_weight)
        
        lm_head_weight = self.weights["lm_head.weight"]
        logits = torch.matmul(x, lm_head_weight.T)
        
        if "lm_head.bias" in self.weights:
            logits = logits + self.weights["lm_head.bias"]
            
        return logits
    
    ## generate contains both greedy decoding as well as sampling
    def generate(self, prompt, max_length=512, do_sample=False, temperature=0.7, top_p=0.95, top_k=50, repetition_penalty=1.2):
        """
        The function generates text based on a given prompt using a language model with options for
        sampling and temperature control.
        
        :param prompt: The `generate` function you provided is used for generating text based on a given
        prompt using a pre-trained language model. Here's an explanation of the parameters used in the
        function:
        :param max_length: The `max_length` parameter in the `generate` function specifies the maximum
        length of the generated text in terms of the number of tokens. The function will stop generating
        text once the length reaches this specified maximum length, defaults to 512 (optional)
        :param do_sample: The `do_sample` parameter in the `generate` function determines whether
        sampling should be used during text generation. If set to `True`, the model will sample the next
        token based on the predicted probabilities, allowing for more diverse and less deterministic
        outputs. If set to `False`, the model will simply, defaults to False (optional)
        :param temperature: The `temperature` parameter in the `generate` function controls the level of
        randomness in the sampling process. A higher temperature value results in more diverse and
        random outputs, while a lower temperature value leads to more conservative and predictable
        outputs. It essentially scales the logits before applying the softmax function during sampling
        :param top_p: The `top_p` parameter in the `generate` function controls the nucleus sampling
        technique. It specifies the cumulative probability threshold in the range [0, 1] for nucleus
        sampling. During sampling, the model considers the most probable tokens whose cumulative
        probability mass exceeds this threshold
        :param top_k: The `top_k` parameter in the `generate` function controls the number of highest
        probability vocabulary tokens to keep for the next word generation. It limits the sampling to
        the top `k` most likely tokens at each step. This can help in preventing the model from
        considering unlikely or rare tokens during generation, defaults to 50 (optional)
        :param repetition_penalty: The `repetition_penalty` parameter in the `generate` function is used
        to penalize the likelihood of generating tokens that have already been generated in the input
        sequence. It helps in reducing repetitive outputs in the generated text
        :return: The `generate` method returns the generated text based on the provided prompt using the
        model. It generates text up to a maximum length, considering parameters like sampling,
        temperature, top-k, top-p, and repetition penalty. The generated text is decoded and returned as
        a string without special tokens.
        """
        self.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        eos_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :]

                for token_id in set(input_ids.view(-1).tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty

                if do_sample:
                    next_token_logits = next_token_logits / temperature

                    if top_k > 0:
                        top_k_vals, _ = torch.topk(next_token_logits, top_k)
                        kth_value = top_k_vals[:, -1].unsqueeze(-1)
                        next_token_logits = torch.where(
                            next_token_logits < kth_value,
                            torch.full_like(next_token_logits, -float('inf')),
                            next_token_logits
                        )

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[:, indices_to_remove] = -float('inf')

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == eos_token_id:
                    break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
