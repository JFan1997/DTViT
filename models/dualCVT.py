from transformers import CvtPreTrainedModel, CvtModel, CvtConfig
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from typing import Optional, Union, Tuple
import torch
class CvtForDualClassification(CvtPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels_task1 = 2  # 第一个分类任务的类别数量
        self.num_labels_task2 = 4  # 第二个分类任务的类别数量
        # self.cvt = CvtModel(config, add_pooling_layer=False)
        self.cvt=CvtModel.from_pretrained("microsoft/cvt-21-384-22k")
        self.layernorm = nn.LayerNorm(config.embed_dim[-1])

        # 两个分类器头
        self.classifier_task1 = nn.Linear(config.embed_dim[-1], self.num_labels_task1)
        self.classifier_task2 = nn.Linear(config.embed_dim[-1], self.num_labels_task2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels_task1: Optional[torch.Tensor] = None,
        labels_task2: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取序列输出和CLS token
        sequence_output = outputs[0]
        cls_token = outputs[1]

        # LayerNorm归一化
        batch_size, num_channels, height, width = sequence_output.shape
        sequence_output = sequence_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        sequence_output = self.layernorm(sequence_output)

        # 平均池化特征图
        sequence_output_mean = sequence_output.mean(dim=1)

        # 通过两个分类头得到两个任务的logits
        logits_task1 = self.classifier_task1(sequence_output_mean)
        logits_task2 = self.classifier_task2(sequence_output_mean)

        loss = None
        if labels_task1 is not None and labels_task2 is not None:
            # 使用交叉熵损失函数计算两个任务的损失
            loss_fct = CrossEntropyLoss()
            loss_task1 = loss_fct(logits_task1.view(-1, self.num_labels_task1), labels_task1.view(-1))
            loss_task2 = loss_fct(logits_task2.view(-1, self.num_labels_task2), labels_task2.view(-1))
            loss = loss_task1 + loss_task2

        if not return_dict:
            output = (logits_task1, logits_task2) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=(logits_task1, logits_task2),
            hidden_states=outputs.hidden_states,
        )

if __name__ == "__main__":
    config = CvtConfig.from_pretrained("microsoft/cvt-21-384-22k")

    model = CvtForDualClassification(config)
    # print(model)
    input = torch.randn(1, 3, 224, 224)

    output = model(input)
    print(output.logits[0].shape, output.logits[1].shape)
    # print(output[0],len(output))
    # print(output[0].shape, output[1].shape)

