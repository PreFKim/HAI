


# LR 실험 1e-4가 제일 베스트 (38)과 비교
# python train_score.py --image_width 256 --image_height 192 --exp_name 41 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type r152 --batch_size 32 --learning_rate 1e-3 --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 42 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type r152 --batch_size 32 --learning_rate 1e-5 --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 43 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type r152 --batch_size 32 --learning_rate 5e-4 --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# rotation 최대 180 -> 45도 ( 45번 비교 )
# python train_score.py --image_width 256 --image_height 192 --exp_name 46 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5  --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# color shift (46번 비교)
# python train_score.py --image_width 256 --image_height 192 --exp_name 47 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5  --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# rotate 최대 180에 0.1 확률 (47번 비교)
# python train_score.py --image_width 256 --image_height 192 --exp_name 49 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5  --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# 색 증강 먼저
# python train_score.py --image_width 256 --image_height 192 --exp_name 51 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type r152 --batch_size 32 --det_crop --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 53 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type r152 --batch_size 32 --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name 55 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type r152 --batch_size 32 --erase_other_car --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature

# python train_score.py --image_width 256 --image_height 192 --exp_name ib50 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type ib --batch_size 32 --det_crop --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature
# python train_score.py --image_width 256 --image_height 192 --exp_name fb50 --epoch 50 --gpu 1 --fold_idx -1 --backbone_type fb --batch_size 32 --det_crop --normalize_type IN --mix_ratio 0.25 --cutmix_ratio 0.5 --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature


# python train_score2.py --exp_name pretrain3 --epoch 100 --gpu 1 --fold_idx 0 --batch_size 512 --backbone_type fs --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature --learning_rate 5e-4
# python train_score2.py --exp_name pretrain4 --epoch 100 --gpu 1 --fold_idx 0 --batch_size 512 --backbone_type fs --out_channels_per_joint 256 --pose_type vec --image_cls_token_per_joint --pose_cls_token_per_joint --temperature --learning_rate 1e-4
python test.py --det_crop --gpu 1 --use_cache
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