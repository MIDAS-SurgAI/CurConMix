# Pre-Training 
python main.py epochs=3 target_size=131 ssl=True exp=CurConMix_Pretraining split_selector='cholect45-crossval' device='cuda:0' method='curriculum_supcon' ssl_loss='supcon' feature_batch=True feature_mixup=True alpha=0.4 label_order='[t,it,ivt]' 

# Fine Tuning 
## Teacher model with Input Mixup 
python main.py epochs=20 target_size=131 exp=Teacher_IM04 pretrained_ssl=True pretrained_exp=CurConMix_Pretraining_2 method='supcon' split_selector='cholect45-crossval' device='cuda:0' mixup=True alpha=0.4 
python generate.py target_size=131 exp=Teacher_IM04 split_selector='cholect45-crossval' inference=False device='cuda:0'  
python main.py epochs=40 target_size=131 exp=Student_IM04 distill=True split_selector='cholect45-crossval' teacher_exp=Teacher_IM04 device='cuda:0' mixup=True alpha=0.4 

python generate.py target_size=131 exp=Teacher_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0'
python generate.py target_size=131 exp=Student_IM04 split_selector='cholect45-crossval' inference=True device='cuda:0'

python evaluate.py inference=true 