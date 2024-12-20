import torch
import torch.nn as nn
import torch.optim as optim

class PntTransformer(nn.Module):
    # Assume the number of points stay the same
    def __init__(self, conf):
        super().__init__()
        #print(conf)
        self.multires = conf.get_int('multires')
        self.input_dim = conf.get_int('input_dim')
        self.output_dim = conf.get_int('output_dim')
        self.embedder, self.pos_encoder_out_dim = self.get_embedder(self.multires, self.input_dim)
        #print(f"pos_encoder_out_dim: {self.pos_encoder_out_dim}")
        self.transformers = nn.Transformer(d_model=self.pos_encoder_out_dim, nhead=self.input_dim)
        self.out = nn.Linear(self.pos_encoder_out_dim,self.output_dim)

    def get_embedder(self, multires, input_dims=3, i=0):
        if i == -1:
            return nn.Identity(), input_dims
        #print(f"calling get_embedder!\n\tmultires: {type(multires)}:{multires}\n\tinput_dims: {input_dims}")
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : input_dims,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
        
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        return embed, embedder_obj.out_dim

    def forward(self, pnt_cloud, time):
        #print(f"pnt_cloud: {pnt_cloud.shape}")
        pnt_cloud_embed = self.embedder(pnt_cloud)
        #print(f"pnt_cloud_embed: {pnt_cloud_embed.shape}")
        time_flat = torch.full_like(pnt_cloud, time.item())
        #print(f"time_flat: {time_flat.shape}")
        time_embed = self.embedder(time_flat)
        #print(f"time_embed: {time_embed.shape}")

        output = self.transformers(pnt_cloud_embed, time_embed)
        return self.out(output)
        


# Positional encoding (section 5.1):
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            #print(f"out_dim: {out_dim}->{out_dim+d} (include_input)")
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                #print(f"out_dim: {out_dim}->{out_dim+d} (freq:{freq})")
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)



