# CurConMix: A Curriculum Contrastive Learning Framework for Enhancing Surgical Action Triplet Recognition

📌 MICCAI 2025 Early Accepted

This work has been Early Accepted to MICCAI 2025, placing in the top 9% of submissions.
![CurConMix Diagram](assets/Figure.png)

🧠 Abstract

Accurately recognizing surgical action triplets in surgical videos is crucial for advancing context-aware systems that deliver real-time feedback, enhancing surgical safety and efficiency. However, recognizing surgical action triplets <instrument, verb, target> is challenging due to subtle variations, complex interdependencies, and severe class imbalance. Most existing approaches focus on individual triplet components while overlooking their interdependencies and the inherent class imbalance in triplet distributions.

To address these challenges, we propose a novel framework, Curriculum Contrastive Learning with Feature Mixup (CurConMix). During pre-training, we employ curriculum contrastive learning, which progressively captures relationships among triplet components and distinguishes fine-grained variations through hard pair sampling and synthetic hard negative generation. In the fine-tuning stage, we further refine the model using self-distillation and mixup strategies to alleviate class imbalance.

We evaluate our framework on the CholecT45 dataset using 5-fold cross-validation. Experimental results demonstrate that our approach surpasses existing methods across various model sizes and input resolutions. Moreover, our findings underscore the importance of capturing interdependency among triplet components, highlighting the effectiveness of our proposed framework in addressing key challenges in surgical action recognition.

🏋️‍♂️ How to Train

Run the following script to perform the entire training pipeline, which includes both pre-training with curriculum contrastive learning and fine-tuning with self-distillation and mixup:

bash train.sh

📦 Model Checkpoints

Model checkpoints will be available here:

🔗 Model Checkpoint Download (Coming soon)

✍️ Citation (Coming soon)

If you use our code or models, please consider citing:

@article{curconmix2025,
  title     = {CurConMix: A Curriculum Contrastive Learning Framework for Enhancing Surgical Action Triplet Recognition},
  author    = {...},
  journal   = {MICCAI},
  year      = {2025},
  note      = {Early Accepted}
}

For questions or issues, please contact the authors or open an issue on GitHub.

