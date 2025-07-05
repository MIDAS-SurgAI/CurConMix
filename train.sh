# ## Curriculum Supcon with Hard pair sampling and feature mixup 
python main.py epochs=3 target_size=131 ssl=True exp=SwinB_Curriculum_Feature_Mixup split_selector='cholect45-crossval' device='cuda:0' method='curriculum_supcon' ssl_loss='supcon' label_order='[t,it,ivt]' feature_batch=True feature_mixup=True alpha=0.4

## Fine Tuning with mixup 
python main.py epochs=20 target_size=131 exp=SwinB_Curriculum_Supcon_Feature_Mixup_Teacher pretrained_ssl=True pretrained_exp=SwinB_Curriculum_Feature_Mixup_2 method='supcon' split_selector='cholect45-crossval' device='cuda:0' mixup=True alpha=0.4
python generate.py target_size=131 exp=SwinB_Curriculum_Supcon_Feature_Mixup_Teacher split_selector='cholect45-crossval' inference=False device='cuda:0' 
python main.py epochs=40 target_size=131 exp=SwinB_Curriculum_Supcon_Feature_Mixup_Student distill=True split_selector='cholect45-crossval' teacher_exp=SwinB_Curriculum_Supcon_Feature_Mixup_Teacher device='cuda:0' mixup=True alpha=0.4 
python generate.py target_size=131 exp=SwinB_Curriculum_Supcon_Feature_Mixup_Student split_selector='cholect45-crossval' inference=True device='cuda:0' 