from torch import nn

class sentiment_model(nn.Module):
    def __init__(self, model,p_val,hidden, augment):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(p=p_val)
        if augment:
            new_classifier = nn.Sequential(
                nn.Linear(model.classifier.in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 3)  # number of the labels
            )
            self.model.classifier = new_classifier

    def forward(self, input_ids, attention_mask, drop_out):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        if drop_out:
            outputs = self.dropout(outputs)
        return outputs


