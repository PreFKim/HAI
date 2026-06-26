

# Det_crop, erase_other_car 실험
# python train_score.py --image_width 256 --image_height 192 --exp_name 38 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 39 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 40 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5  --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# rotate 넓이 제한 + 0.1 -> 0.25 ( 40과 비교 )
# python train_score.py --image_width 256 --image_height 192 --exp_name 44 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5  --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# 증강 순서 변경 색 -> 공간 증강 (44와 비교)
# python train_score.py --image_width 256 --image_height 192 --exp_name 45 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5  --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# gauss noise (47번 비교)
# python train_score.py --image_width 256 --image_height 192 --exp_name 48 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5  --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature


# 공간증강 먼저
# python train_score.py --image_width 256 --image_height 192 --exp_name 50 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 52 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 54 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# python train_score.py --image_width 256 --image_height 192 --exp_name fb100 --epoch 100 --gpu 0 --fold_idx -1 --backbone_type fb --batch_size 32 --det_crop --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 54 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name is50 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type is --batch_size 32 --det_crop --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature


# Rotate 비교 45일때가 제일 베스트
# # rotate 90
# python train_score.py --image_width 256 --image_height 192 --exp_name 56 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# # rotate 45
# python train_score.py --image_width 256 --image_height 192 --exp_name 57 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# # # rotate 30
# # python train_score.py --image_width 256 --image_height 192 --exp_name 58 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# rotate 90 0.01 
# python train_score.py --image_width 256 --image_height 192 --exp_name 59 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# internimage learning rate 비교 (is50이랑 비교) lr 5e-4가 제일 베스트
# python train_score.py --image_width 256 --image_height 192 --exp_name 61 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type is --learning_rate 5e-4 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 60 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type is --learning_rate 5e-4 --batch_size 32 --det_crop --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature #internimage는 
# python train_score.py --image_width 256 --image_height 192 --exp_name 62 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type is --learning_rate 1e-3 --batch_size 32 --det_crop --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature #internimage는 


# erase car -> space -> color (57이랑 비교))
# python train_score.py --image_width 256 --image_height 192 --exp_name 63 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# erase car ( 0.5 ) -> space -> color (57이랑 비교))  성능 향상이 별로 안된다 Erase Car 때문이 아닌가?
# python train_score.py --image_width 256 --image_height 192 --exp_name 64 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# erase car가 dropout의 역할을 해줘서 그렇다면? -> Coarse를 crop 이후에 넣자(dropout 이후 crop을 한다면 dropout이 없는 부분일 수 있음) -> 성능 향상
# python train_score.py --image_width 256 --image_height 192 --exp_name 65 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# Dropout의 크기를 키워보자 -> 성능 감소
# python train_score.py --image_width 256 --image_height 192 --exp_name 66 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# rotate 0.25 (65랑 비교) 
# python train_score.py --image_width 256 --image_height 192 --exp_name 67 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# crop 0.5 (57이랑 비교) 
# crop 0.5(random 0.5 half 0.5) (57이랑 비교) 
# python train_score.py --image_width 256 --image_height 192 --exp_name 66 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 67 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# crop rotate 0.5 (57이랑 비교) 
# python train_score.py --image_width 256 --image_height 192 --exp_name 70 --epoch 50 --gpu 0 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature


# 최종
# python train_score.py --exp_name pretrain --epoch 100 --gpu 0 --fold_idx 0 --batch_size 512 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature



# 384 512 테스트
# r152 4분
# it 7분
# is 12분 
# ib 19분
# il 48분
# ixl 1시간
# cb 17분

# 192 256 테스트
# r125 1분 30초
# it 2분 10초
# is 3분
# ib 5분
# cs 2분
# cb 4분 30초

# vit
# vn 3분 30초
# vc 14분
# st 2분
# ss 3분 30초
# sb 19분