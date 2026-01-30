"""3-class classification model for gripper control.

Predicts gripper state transitions:
0: KEEP_CURRENT - 保持当前状态
1: TRIGGER_CLOSE - OPEN→CLOSE（抓取时刻）
2: TRIGGER_OPEN - CLOSE→OPEN（开始接近）
"""

import torch
import torch.nn as nn


class GripperPredictor(nn.Module):
    """3-class classifier for gripper state transitions.
    
    Input: [gripper_state(1), ee_pos(3), object_pos(3)] = 7
    Output: logits [3] -> softmax -> [prob_keep, prob_close, prob_open]
    """
    
    def __init__(self, hidden_dims=[128, 128, 64], dropout=0.1):
        """Initialize GripperPredictor.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        input_dim = 7  # gripper_state(1) + ee_pos(3) + object_pos(3)
        output_dim = 3  # 3分类：KEEP_CURRENT (0), TRIGGER_CLOSE (1), TRIGGER_OPEN (2)
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层：3个logits
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, gripper_state, ee_pos, object_pos):
        """Forward pass.
        
        Args:
            gripper_state: [B, 1] - Current gripper joint value
            ee_pos: [B, 3] - End-effector position
            object_pos: [B, 3] - Object position
            
        Returns:
            logits: [B, 3] - raw logits for [KEEP_CURRENT, TRIGGER_CLOSE, TRIGGER_OPEN]
        """
        x = torch.cat([gripper_state, ee_pos, object_pos], dim=-1)  # [B, 7]
        logits = self.network(x)  # [B, 3]
        return logits
    
    def predict(self, gripper_state, ee_pos, object_pos):
        """Predict gripper action (transition detection).
        
        Args:
            gripper_state: [B, 1] - Current gripper joint value
            ee_pos: [B, 3] - End-effector position
            object_pos: [B, 3] - Object position
        
        Returns:
            action: [B, 1] - 1.0 for open, -1.0 for close
            confidence: [B, 1] - confidence (probability of predicted class)
            pred_class: [B, 1] - predicted class (0=KEEP_CURRENT, 1=TRIGGER_CLOSE, 2=TRIGGER_OPEN)
        """
        logits = self.forward(gripper_state, ee_pos, object_pos)  # [B, 3]
        probs = torch.softmax(logits, dim=-1)  # [B, 3]
        
        # Class 0 = KEEP_CURRENT, Class 1 = TRIGGER_CLOSE, Class 2 = TRIGGER_OPEN
        pred_class = torch.argmax(probs, dim=-1, keepdim=True)  # [B, 1]
        
        # Map to action: TRIGGER_CLOSE -> -1.0 (close), others -> 1.0 (open/keep)
        action = torch.where(
            pred_class == 1,  # TRIGGER_CLOSE
            torch.full_like(pred_class, -1.0, dtype=torch.float32),  # CLOSE
            torch.ones_like(pred_class, dtype=torch.float32)  # OPEN/KEEP
        )
        
        # Confidence = probability of predicted class
        confidence = torch.gather(probs, 1, pred_class)  # [B, 1]
        
        return action, confidence, pred_class
