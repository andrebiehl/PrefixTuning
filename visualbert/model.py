import torch
import torch.nn as nn
from transformers import VisualBertPreTrainedModel, VisualBertModel

class VisualBERTCaptionGenerator(VisualBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual_bert = VisualBertModel(config)
        self.prefix_dim = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()

    def initialize_prefix(self, prefix_length):
        self.prefix_length = prefix_length
        self.prefix_embeddings = nn.Parameter(torch.zeros(prefix_length, self.prefix_dim))
        self.prefix_proj = nn.Sequential(
            nn.Linear(self.prefix_dim, self.prefix_dim),
            nn.ReLU(),
            nn.Linear(self.prefix_dim, self.prefix_length * 2 * self.config.num_hidden_layers * self.prefix_dim)
        )
    
    def get_prompt(self, batch_size):
        prefix_embeddings = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return self.prefix_proj(prefix_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        batch_size = input_ids.shape[0]
        prefix_embeddings = self.get_prompt(batch_size)
    
        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        sequence_output = outputs[0]
        sequence_output = torch.cat((prefix_embeddings, sequence_output), dim=1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
    
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    
        return {'loss': loss, 'logits': logits}
