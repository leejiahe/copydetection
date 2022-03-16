# Inspired bny
# https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/vit/modeling_vit.py

import math

import torch
import torch.nn as nn

from patch_embeddings import PatchEmbeddings

class CopyDetectEmbedding(nn.Module):
    def __init__(self,
                 config: object,
                 vit_cls: torch.Tensor,
                 pos_emb: torch.Tensor):
        super().__init__()
        self.vit_cls = vit_cls #pretrained cls token from ViT model
        self.position_embeddings = pos_emb #pretrained positional embedding from ViT model
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) #cls token for the similar image prediction
        self.sep_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) #sep token to separate sequence of the two images
        self.img_segment = nn.Embedding(2, config.hidden_size) #image segment to differentiate image r and image q

        self.patch_embeddings = PatchEmbeddings(image_size = config.image_size,
                                                patch_size = config.patch_size,
                                                num_channels = config.num_channels,
                                                embed_dim = config.hidden_size)
        
        self.patch_size = config.patch_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_pos_encoding(self, img, interpolate_pos_encoding = False):
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        batch_size, num_channels, height, width = img.shape
        embeddings = self.patch_embeddings(img, interpolate_pos_encoding = interpolate_pos_encoding)

        # add the pretrained [CLS] token to the embedded patch tokens
        vit_cls = self.vit_cls.expand(batch_size, -1, -1)
        embeddings = torch.cat((vit_cls, embeddings), dim = 1)
        
        npatch = embeddings.shape[1] - 1
        N = self.position_embeddings.shape[1] - 1
        if npatch == N and height == width:
            embeddings = embeddings + self.position_embeddings
            return self.dropout(embeddings)
        
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        
        h0 = height // self.patch_size
        w0 = width // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                                                    scale_factor = (h0 / math.sqrt(N), w0 / math.sqrt(N)), mode = "bicubic", align_corners = False)
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        embeddings = embeddings + torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim = 1)
        
        return self.dropout(embeddings)
    
    def forward(self, img_r, img_q = None, interpolate_pos_encoding = False):
        emb_r = self.get_pos_encoding(img_r, interpolate_pos_encoding)
        if img_q is None:
            return emb_r            
        else:
            emb_q = self.get_pos_encoding(img_q, interpolate_pos_encoding)
            batch_size, seq_len_r, _ = emb_r.shape # shape: batch_size, seq_len, dim
            # First image segment (similar to sentence A in NSP) 
            segment_r = self.img_segment(torch.zeros((batch_size, seq_len_r), device = self.img_segment.weight.device, dtype = torch.int))
            # Second image segment (similar to sentence B in NSP)
            segment_q = self.img_segment(torch.ones((batch_size, emb_q.size(1)), device = self.img_segment.weight.device, dtype = torch.int))
            # Add segment embedding to reference and query embeddings
            emb_seg_r = emb_r + segment_r
            emb_seg_q = emb_q + segment_q
            # Added first segment (similar to NSP) to cls and sep tokens, there is not positional encoding for this two tokens
            cls_token = self.cls_token.expand(batch_size, -1, -1) + segment_r 
            sep_token = self.sep_token.expand(batch_size, -1, -1) + segment_r
            # Concat cls, ref emb, sep, query emb
            emb_rq = torch.cat((cls_token, emb_seg_r, sep_token, emb_seg_q), dim = 1)
            return emb_r, emb_q, emb_rq
