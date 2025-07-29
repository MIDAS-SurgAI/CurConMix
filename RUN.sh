
# # ## Curriculum Learning with Pos/Negative Sampling and feature mixup
# python main.py epochs=3 target_size=131 ssl=True exp=CurConMix_e3 split_selector='cholect45-crossval' cutout=False device='cuda:0' method='curriculum_supcon' ssl_loss='supcon' feature_batch=True feature_mixup=True alpha=0.4 label_order='[t,it,ivt]' ## 20250721

# ## Fine Tuning 
# python main.py epochs=20 target_size=131 exp=20250721_CurConMix_Teacher_IM04 pretrained_ssl=True pretrained_exp=CurConMix_e3_2 method='supcon' split_selector='cholect45-crossval' device='cuda:0' mixup=True alpha=0.4 ## 20250721
# python generate.py target_size=131 exp=20250721_CurConMix_Teacher_IM04 split_selector='cholect45-crossval' inference=False device='cuda:0' #20250721 
# python main.py epochs=40 target_size=131 exp=20250721_CurConMix_Student_IM04 distill=True split_selector='cholect45-crossval' teacher_exp=20250721_CurConMix_Teacher_IM04 device='cuda:0' mixup=True alpha=0.4 #20250721

# python generate.py target_size=131 exp=20250721_CurConMix_Teacher_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0' #20250721 
# python generate.py target_size=131 exp=20250721_CurConMix_Student_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0' #20250721 

# # ## Curriculum Learning with Pos/Negative Sampling and feature mixup
# python main.py epochs=3 target_size=131 ssl=True exp=CurConMix_e3_using_new_matrix split_selector='cholect45-crossval' cutout=False device='cuda:0' method='curriculum_supcon' ssl_loss='supcon' feature_batch=True feature_mixup=True alpha=0.4 label_order='[t,it,ivt]' ## 20250723

# ## Fine Tuning 
# python main.py epochs=20 target_size=131 exp=20250723_CurConMix_Teacher_IM04 pretrained_ssl=True pretrained_exp=CurConMix_e3_using_new_matrix_2 method='supcon' split_selector='cholect45-crossval' device='cuda:0' mixup=True alpha=0.4 ## 20250723
# python generate.py target_size=131 exp=20250723_CurConMix_Teacher_IM04 split_selector='cholect45-crossval' inference=False device='cuda:0' #20250723 
# python main.py epochs=40 target_size=131 exp=20250723_CurConMix_Student_IM04 distill=True split_selector='cholect45-crossval' teacher_exp=20250723_CurConMix_Teacher_IM04 device='cuda:0' mixup=True alpha=0.4 #20250723

# python generate.py target_size=131 exp=20250723_CurConMix_Teacher_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0' #20250723
# python generate.py target_size=131 exp=20250723_CurConMix_Student_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0' #20250723 

# # ## Curriculum Learning with Pos/Negative Sampling and feature mixup 한번 더 
# python main.py epochs=3 target_size=131 ssl=True exp=CurConMix_e3_using_new_matrix_again split_selector='cholect45-crossval' device='cuda:0' method='curriculum_supcon' ssl_loss='supcon' feature_batch=True feature_mixup=True alpha=0.4 label_order='[t,it,ivt]' ## 20250724

# ## Fine Tuning 
# python main.py epochs=20 target_size=131 exp=20250724_CurConMix_Teacher_IM04 pretrained_ssl=True pretrained_exp=CurConMix_e3_using_new_matrix_again_2 method='supcon' split_selector='cholect45-crossval' device='cuda:0' mixup=True alpha=0.4 ## 20250724
# python generate.py target_size=131 exp=20250724_CurConMix_Teacher_IM04 split_selector='cholect45-crossval' inference=False device='cuda:0' #20250724 
# python main.py epochs=40 target_size=131 exp=20250724_CurConMix_Student_IM04 distill=True split_selector='cholect45-crossval' teacher_exp=20250724_CurConMix_Teacher_IM04 device='cuda:0' mixup=True alpha=0.4 #20250724

# python generate.py target_size=131 exp=20250724_CurConMix_Teacher_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0' #20250724
# python generate.py target_size=131 exp=20250724_CurConMix_Student_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0' #20250724 


python main.py epochs=3 target_size=131 ssl=True exp=20250728_CurConMix_e3 split_selector='cholect45-crossval' cutout=False device='cuda:0' method='curriculum_supcon' ssl_loss='supcon' feature_batch=True feature_mixup=True alpha=0.4 label_order='[t,it,ivt]' ## 20250721

# ## Fine Tuning 
python main.py epochs=20 target_size=131 exp=20250728_CurConMix_Teacher_IM04 pretrained_ssl=True pretrained_exp=20250728_CurConMix_e3_2 method='supcon' split_selector='cholect45-crossval' device='cuda:0' mixup=True alpha=0.4 ## 20250721
python generate.py target_size=131 exp=20250728_CurConMix_Teacher_IM04 split_selector='cholect45-crossval' inference=False device='cuda:0' #20250721 
python main.py epochs=40 target_size=131 exp=20250728_CurConMix_Student_IM04 distill=True split_selector='cholect45-crossval' teacher_exp=20250728_CurConMix_Teacher_IM04 device='cuda:0' mixup=True alpha=0.4 #20250721

python generate.py target_size=131 exp=20250728_CurConMix_Teacher_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0' #20250721 
python generate.py target_size=131 exp=20250728_CurConMix_Student_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0' #20250721 