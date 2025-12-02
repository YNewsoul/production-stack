# 单个请求预测示例 - predict_single_request.py
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
import os

class BertRegressionModel(torch.nn.Module):
    def __init__(self, config, model_name, hidden_dim):
        super().__init__()
        self.config = config
        self.bert = transformers.BertModel.from_pretrained(model_name, force_download=False)
        # 固定预训练模型权重
        for param in self.bert.parameters():
            param.requires_grad = False

        self.cls = torch.nn.Linear(config.hidden_size, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)
        return output

class Predictor:
    def __init__(self, model_path="predictions_llama-13b_warmup_reg_l1_1000K.pth",model_name="bert-base-uncased", device="cuda:0" if torch.cuda.is_available() else "cpu"):
        """初始化预测服务，加载模型到内存"""
        self.device = device
        self.model_path = model_path
        self.model_name = model_name
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=False)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        
        # 加载配置和模型
        self.config = AutoConfig.from_pretrained(model_name, force_download=False)
        
        self.model = BertRegressionModel(self.config, model_name, hidden_dim=128).to(device)
        
        # 加载模型权重
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, prompt):
        """对单个提示进行预测"""
        
        # 处理输入文本
        inputs = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 如果输入长度超过512，只保留尾部512个token
        if input_ids.shape[1] >= 512:
            input_ids = input_ids[:, -512:]
            attention_mask = attention_mask[:, -512:]
        
        # 移至设备
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # 进行预测
        with torch.no_grad():
            prediction = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            predicted_length = prediction.item()
        
        return int(predicted_length)